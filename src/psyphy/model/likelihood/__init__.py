"""psyphy.model.likelihood

Task likelihoods for psychophysical experiments: the mapping from a model and a
set of stimuli to the likelihood of an observed response.

Likelihoods are grouped by response distribution. The log-likelihood is written
once per distribution, and a new task only implements ``predict``:

Modules
-------
base     : TaskLikelihood, BernoulliTaskLikelihood, GaussianTaskLikelihood
oddity   : OddityTaskConfig, OddityTask (3-AFC, Monte Carlo)
features : whiten, reduce_2d, whitened_oddity_features, the inputs a neural
           surrogate takes
neural   : NeuralOddityTask, NeuralSurrogate and the weight-file format

Every public name is re-exported here, so existing imports are unaffected::

    from psyphy.model.likelihood import OddityTask, TaskLikelihood

Choosing an oddity likelihood
-----------------------------
:class:`OddityTask` and :class:`NeuralOddityTask` model the same observer and the
same decision rule. ``OddityTask`` simulates noisy trials and counts how often the
rule picks the odd stimulus. ``NeuralOddityTask`` evaluates a small network fitted
to that simulation in advance.

| | ``OddityTask`` | ``NeuralOddityTask`` |
|---|---|---|
| how ``P(correct)`` is computed | Monte Carlo simulation | a fitted network |
| stimulus dimensions | any | 2 |
| noise models | any | ``GaussianNoise`` |
| same value for every PRNG key | no | yes |
| cost per evaluation | grows with ``num_samples`` | one forward pass |
| weight file | none | ships with psyphy |

Use :class:`NeuralOddityTask` for a 2-D model with Gaussian noise, especially
when the likelihood is differentiated many times, as in NUTS or an optimizer.
Use :class:`OddityTask` for any other dimension or noise model, and to check a
surrogate fit against the simulation it was trained on.
"""

from .base import (
    BernoulliTaskLikelihood,
    GaussianTaskLikelihood,
    TaskLikelihood,
)
from .features import reduce_2d, whiten, whitened_oddity_features
from .neural import (
    NeuralOddityTask,
    NeuralSurrogate,
    mlp_forward,
    save_surrogate_npz,
)
from .oddity import OddityTask, OddityTaskConfig

__all__ = [
    # Base hierarchy
    "TaskLikelihood",
    "BernoulliTaskLikelihood",
    "GaussianTaskLikelihood",
    # Concrete tasks
    "OddityTaskConfig",
    "OddityTask",
    # Surrogate feature maps
    "whiten",
    "reduce_2d",
    "whitened_oddity_features",
    # Trained surrogates and their file format
    "NeuralOddityTask",
    "NeuralSurrogate",
    "mlp_forward",
    "save_surrogate_npz",
]
