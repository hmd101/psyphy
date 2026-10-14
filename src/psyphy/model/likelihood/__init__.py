"""psyphy.model.likelihood

Task likelihoods for psychophysical experiments: the mapping from a model and a
set of stimuli to the likelihood of an observed response.

Organised by the assumed response distribution rather than by task, so that the
log-likelihood is written once per distribution and a new task only implements
``predict``:

Modules
-------
base     : TaskLikelihood, BernoulliTaskLikelihood, GaussianTaskLikelihood
oddity   : OddityTaskConfig, OddityTask (3-AFC, Monte Carlo)
features : whiten, reduce_2d, whitened_oddity_features -- the input transform a
           neural surrogate consumes, public because deriving one for a new task
           is something users are meant to be able to do

Every public name is re-exported here, so existing imports are unaffected::

    from psyphy.model.likelihood import OddityTask, TaskLikelihood
"""

from .base import (
    BernoulliTaskLikelihood,
    GaussianTaskLikelihood,
    TaskLikelihood,
)
from .features import reduce_2d, whiten, whitened_oddity_features
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
]
