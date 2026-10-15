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
neural   : NeuralOddityTask, NeuralSurrogate and the on-disk weight format. A
           surrogate file records which task, dimension and feature map it was
           trained for, because the weights alone cannot say, and loading the
           wrong ones fails silently rather than loudly

Every public name is re-exported here, so existing imports are unaffected::

    from psyphy.model.likelihood import OddityTask, TaskLikelihood

Two oddity likelihoods: which to use
------------------------------------
:class:`OddityTask` and :class:`NeuralOddityTask` describe the *same* observer
and the *same* decision rule. They differ only in how ``P(correct)`` is obtained:
the first simulates noisy trials and counts how often the rule succeeds, the
second evaluates a small network that was fitted to that simulation in advance.

================================  =====================  =========================
                                  ``OddityTask``         ``NeuralOddityTask``
================================  =====================  =========================
How ``P(correct)`` is obtained    Monte Carlo simulation  a fitted network
Stimulus dimensions               any                     2 only
Noise models                      any                     Gaussian only
Same answer on every call         no                      yes
Cost per likelihood evaluation    grows with sample count fixed, and much smaller
Needs a weight file               no                      yes, one ships with psyphy
================================  =====================  =========================

**The Monte Carlo task is the definition; the surrogate is an approximation of
it.** That is worth stating plainly, because it explains several things that
would otherwise look arbitrary: the network was trained on ``OddityTask``'s
output, the test suite checks the two against each other rather than against any
external ground truth, and ``OddityTask`` is not going away.

Use :class:`NeuralOddityTask` when the model is two-dimensional with Gaussian
noise, which covers the common case, and especially when the likelihood is being
differentiated repeatedly. Beyond the speed, it returns the same value every
time it is called, and some methods need that: gradient-based samplers assume the
log density is a fixed function of the parameters, and an approximation built
from second derivatives is meaningless when the first ones are noisy.

Use :class:`OddityTask` for anything else: a different dimensionality, a
different noise model, or when you want the simulation itself rather than a
fitted stand-in for it.

Accuracy is not the deciding factor in either direction. At the sample counts
normally used, the surrogate sits closer to a high-accuracy simulation than the
simulation does at its own defaults, so choosing it is not a trade of accuracy
for speed.
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
