"""psyphy.model.likelihood

Task likelihoods for psychophysical experiments: the mapping from a model and a
set of stimuli to the likelihood of an observed response.

Likelihoods are grouped by response distribution. The log-likelihood is written
once per distribution, and a new task only implements ``predict``:

Modules
-------
base     : TaskLikelihood, BernoulliTaskLikelihood, GaussianTaskLikelihood,
           log_probs_from_logit
oddity   : OddityTaskConfig, OddityTask (3-AFC, Monte Carlo)
features : normalize_by_noise, align_displacement_2d, oddity_features, the inputs a neural
           network takes
neural   : FastOddityTask, TrainedNetwork and the weight-file format

Every public name is re-exported here, so existing imports are unaffected::

    from psyphy.model.likelihood import OddityTask, TaskLikelihood

Choosing an oddity likelihood
-----------------------------
:class:`OddityTask` and :class:`FastOddityTask` model the same observer and the
same decision rule. ``OddityTask`` simulates noisy trials and counts how often the
rule picks the odd stimulus. ``FastOddityTask`` evaluates a small network fitted
to that simulation in advance.

| | ``OddityTask`` | ``FastOddityTask`` |
|---|---|---|
| how ``P(correct)`` is computed | Monte Carlo simulation | a fitted network |
| stimulus dimensions | any | 2 |
| noise models | any | ``GaussianNoise`` |
| same value for every PRNG key | no | yes |
| cost per evaluation | grows with ``num_samples`` | one forward pass |
| weight file | none | ships with psyphy |

Use :class:`FastOddityTask` for a 2-D model with Gaussian noise, especially
when the likelihood is differentiated many times, as in NUTS or an optimizer.
Use :class:`OddityTask` for any other dimension or noise model, and to check a
network fit against the simulation it was trained on.
"""

from .base import (
    BernoulliTaskLikelihood,
    GaussianTaskLikelihood,
    TaskLikelihood,
    log_probs_from_logit,
)
from .features import align_displacement_2d, normalize_by_noise, oddity_features
from .neural import (
    FastOddityTask,
    TrainedNetwork,
    forward_pass,
    save_network,
)
from .oddity import OddityTask, OddityTaskConfig

__all__ = [
    # Base hierarchy
    "TaskLikelihood",
    "BernoulliTaskLikelihood",
    "GaussianTaskLikelihood",
    "log_probs_from_logit",
    # Concrete tasks
    "OddityTaskConfig",
    "OddityTask",
    # Feature maps
    "normalize_by_noise",
    "align_displacement_2d",
    "oddity_features",
    # Trained networks and their file format
    "FastOddityTask",
    "TrainedNetwork",
    "forward_pass",
    "save_network",
]
