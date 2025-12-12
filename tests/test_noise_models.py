"""
test_noise_models.py
--------------------

Tests for different noise models (Gaussian vs Student-t) in WPPM.
"""

import jax.numpy as jnp
import jax.random as jr
import pytest

from psyphy.data.dataset import ResponseData
from psyphy.model import WPPM, Prior
from psyphy.model.noise import GaussianNoise, StudentTNoise
from psyphy.model.task import OddityTask


class TestNoiseModels:
    """Test that different noise models affect MC likelihood."""

    @pytest.fixture
    def model_gaussian(self):
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, scale=0.5),
            task=OddityTask(slope=1.5),
            noise=GaussianNoise(sigma=0.03),
        )

    @pytest.fixture
    def model_student_t(self):
        return WPPM(
            input_dim=2,
            prior=Prior(input_dim=2, scale=0.5),
            task=OddityTask(slope=1.5),
            noise=StudentTNoise(df=3.0, scale=0.03),
        )

    def test_noise_models_give_different_results(self, model_gaussian, model_student_t):
        """
        Test that Gaussian and Student-t noise produce different likelihoods.

        Student-t has heavier tails, so it should behave differently, especially
        for outliers or when discriminability is marginal.
        """
        params = model_gaussian.init_params(jr.PRNGKey(42))

        # Create a trial
        data = ResponseData()
        data.add_trial(
            ref=jnp.array([0.0, 0.0]), comparison=jnp.array([0.1, 0.1]), resp=1
        )

        # Compute likelihood with Gaussian noise
        ll_gaussian = model_gaussian.task.loglik_mc(
            params=params,
            data=data,
            model=model_gaussian,
            noise=model_gaussian.noise,
            num_samples=2000,
            bandwidth=1e-2,
            key=jr.PRNGKey(0),
        )

        # Compute likelihood with Student-t noise
        ll_student = model_student_t.task.loglik_mc(
            params=params,
            data=data,
            model=model_student_t,
            noise=model_student_t.noise,
            num_samples=2000,
            bandwidth=1e-2,
            key=jr.PRNGKey(0),
        )

        # They should be different
        # Note: We use a large sample size to ensure difference isn't just MC noise
        assert not jnp.isclose(ll_gaussian, ll_student, rtol=1e-3), (
            f"Gaussian ({ll_gaussian}) and Student-t ({ll_student}) gave same likelihood!"
        )

    def test_student_t_heavy_tails(self, model_student_t):
        """
        Test that Student-t noise works and produces finite likelihoods.
        """
        params = model_student_t.init_params(jr.PRNGKey(0))
        data = ResponseData()
        data.add_trial(
            ref=jnp.array([0.0, 0.0]), comparison=jnp.array([0.5, 0.5]), resp=1
        )

        ll = model_student_t.task.loglik_mc(
            params=params,
            data=data,
            model=model_student_t,
            noise=model_student_t.noise,
            num_samples=500,
            bandwidth=1e-2,
            key=jr.PRNGKey(0),
        )

        assert jnp.isfinite(ll)
        assert ll <= 0.0
