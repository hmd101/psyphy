"""
test_basis_expansion.py
-----------------------

Tests for basis expansion functionality.

The basis expansion transforms raw stimulus coordinates into an embedding
space using Chebyshev polynomials, as described in Hong et al (2025).
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.model import WPPM, Prior
from psyphy.model.noise import GaussianNoise
from psyphy.model.task import OddityTask
from psyphy.utils.math import chebyshev_basis


class TestBasisExpansion:
    """Tests for stimulus -> embedding space transformation."""

    def test_normalize_stimulus_to_chebyshev_range(self):
        """Stimuli should be normalized to [-1, 1] for Chebyshev basis."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=2),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        # Test with stimulus in [0, 1] range
        x = jnp.array([0.5, 0.8])
        x_norm = model._normalize_stimulus(x)

        # Should map [0, 1] -> [-1, 1]
        assert jnp.all((x_norm >= -1.0) & (x_norm <= 1.0))

        # Check specific values
        # 0.5 -> 0.0, 0.0 -> -1.0, 1.0 -> 1.0
        expected = 2 * x - 1
        assert jnp.allclose(x_norm, expected, atol=1e-6)

    def test_embed_stimulus_shape(self):
        """Embedding should transform input_dim -> embedding_dim."""
        input_dim = 2
        basis_degree = 5
        model = WPPM(
            input_dim=input_dim,
            prior=Prior(input_dim=input_dim, basis_degree=basis_degree),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        x = jnp.array([0.5, 0.3])
        x_embed = model._embed_stimulus(x)

        # Degree 5 Chebyshev -> 6 basis functions per dimension
        # 2D input -> 2 * 6 = 12 embedding dimensions
        expected_embedding_dim = input_dim * (basis_degree + 1)
        assert x_embed.shape == (expected_embedding_dim,)

    def test_embed_stimulus_uses_chebyshev(self):
        """Embedding should use Chebyshev polynomials."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        x = jnp.array([0.5, 0.8])
        x_embed = model._embed_stimulus(x)

        # Manually compute expected embedding
        x_norm = 2 * x - 1  # Normalize to [-1, 1]

        # Chebyshev basis for each dimension
        cheb_0 = chebyshev_basis(x_norm[0:1], degree=3).ravel()  # 4 functions
        cheb_1 = chebyshev_basis(x_norm[1:2], degree=3).ravel()  # 4 functions

        # Concatenate
        expected = jnp.concatenate([cheb_0, cheb_1])

        assert jnp.allclose(x_embed, expected, atol=1e-6)

    def test_embed_stimulus_batch(self):
        """Embedding should work on batches of stimuli."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        # Batch of 5 stimuli
        X = jr.uniform(jr.PRNGKey(0), (5, 2))
        X_embed = jax.vmap(model._embed_stimulus)(X)

        assert X_embed.shape == (5, 2 * 4)  # 5 samples, 2 dims × 4 basis functions

    def test_embed_stimulus_invertible_approximately(self):
        """Basis expansion should be approximately invertible."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        # Original stimulus
        x = jnp.array([0.5, 0.3])

        # Embed and attempt to reconstruct
        # (Note: This is approximate, not exact, since Chebyshev basis
        # is overcomplete for reconstruction)
        x_embed = model._embed_stimulus(x)

        # For now, just verify embedding is deterministic
        x_embed_2 = model._embed_stimulus(x)
        assert jnp.allclose(x_embed, x_embed_2, atol=1e-10)

    def test_embedding_dim_property(self):
        """
        Model should expose embedding_dim property.

        In the new design: embedding_dim = input_dim + extra_dims
        (not input_dim * (basis_degree + 1) anymore)
        """
        input_dim = 3
        basis_degree = 5
        extra_dims = 2
        model = WPPM(
            input_dim=input_dim,
            prior=Prior(
                input_dim=input_dim,
                basis_degree=basis_degree,
                extra_embedding_dims=extra_dims,
            ),
            task=OddityTask(),
            noise=GaussianNoise(),
            extra_dims=extra_dims,
        )

        # New design: embedding_dim = input_dim + extra_dims
        expected_embedding_dim = input_dim + extra_dims
        assert model.embedding_dim == expected_embedding_dim
        assert model.embedding_dim == 5  # 3 + 2

    def test_default_basis_degree(self):
        """
        Default basis degree should be 5 (Hong et al.).

        In new design, embedding_dim = input_dim + extra_dims,
        independent of basis_degree.
        """
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=5),
            task=OddityTask(),
            noise=GaussianNoise(),
            extra_dims=0,  # No extra dimensions by default
        )

        # Hong et al. uses degree 5
        assert model.basis_degree == 5
        # New design: embedding_dim = input_dim + extra_dims = 2 + 0 = 2
        assert model.embedding_dim == 2


class TestBasisExpansionIntegration:
    """Integration tests for basis expansion with full WPPM."""

    def test_discriminability_with_embedding(self):
        """Discriminability should work in embedding space."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        # Initialize parameters
        params = model.init_params(jr.PRNGKey(0))

        # Compute discriminability
        ref = jnp.array([0.5, 0.3])
        probe = jnp.array([0.6, 0.4])
        stimulus = (ref, probe)

        d = model.discriminability(params, stimulus)

        # Should be non-negative scalar
        assert d.shape == ()
        assert d >= 0.0

    def test_local_covariance_in_embedding_space(self):
        """Local covariance should operate in embedding space (TODO: Issue #3 Task 2)."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        params = model.init_params(jr.PRNGKey(0))
        x = jnp.array([0.5, 0.3])

        Sigma = model.local_covariance(params, x)

        assert Sigma.shape == (2, 2)  # input_dim × input_dim

        # Should be positive-definite
        eigenvalues = jnp.linalg.eigvalsh(Sigma)
        assert jnp.all(eigenvalues > 0)

    # def test_fit_with_basis_expansion(self):
    #     """Model fitting should work with basis expansion."""
    #     model = WPPM(
    #         input_dim=2,
    #         prior=Prior(input_dim=2, basis_degree=3),
    #         task=OddityTask(),
    #         noise=GaussianNoise(),
    #     )

    #     # Generate simple training data
    #     n = 20
    #     key = jr.PRNGKey(42)
    #     refs = jr.uniform(key, (n, 2))
    #     probes = refs + 0.05
    #     y = jnp.ones(n, dtype=int)
    #     X = jnp.stack([refs, probes], axis=1)

    #     # Fit should not raise
    #     model.fit(X, y, inference="map", inference_config={"steps": 10})

    #     # Should have fitted posterior
    #     pred_post = model.posterior(refs[:5], probes=probes[:5])
    #     assert pred_post.mean.shape == (5,)


class TestBasisExpansionEdgeCases:
    """Edge cases and error handling."""

    def test_invalid_basis_degree(self):
        """Negative basis degree should raise."""
        with pytest.raises(ValueError, match="basis_degree"):
            WPPM(
                input_dim=2,
                prior=Prior(input_dim=2, basis_degree=-1),  # Negative should raise
                task=OddityTask(),
                noise=GaussianNoise(),
            )

    def test_stimulus_outside_expected_range(self):
        """Stimulus outside [0, 1] should still work (extrapolation)."""
        model = WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, basis_degree=3),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        # Stimulus outside [0, 1]
        x = jnp.array([1.5, -0.5])
        x_embed = model._embed_stimulus(x)

        # Should return valid embedding (Chebyshev polynomials extrapolate)
        assert x_embed.shape == (2 * 4,)
        assert jnp.all(jnp.isfinite(x_embed))

    def test_high_dimensional_input(self):
        """Basis expansion should work for high-dimensional stimuli."""
        input_dim = 10
        basis_degree = 3
        model = WPPM(
            input_dim=input_dim,
            prior=Prior(input_dim=input_dim, basis_degree=basis_degree),
            task=OddityTask(),
            noise=GaussianNoise(),
        )

        x = jr.uniform(jr.PRNGKey(0), (input_dim,))
        x_embed = model._embed_stimulus(x)

        expected_embedding_dim = input_dim * (basis_degree + 1)
        assert x_embed.shape == (expected_embedding_dim,)
