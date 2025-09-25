
# Minimal end2end example

```python
from psyphy import WPPM, Prior, OddityTask
from psyphy.inference import LangevinSampler
from psyphy.trial_placement.info_gain import InfoGainPlacement
from psyphy.session import ExperimentSession
from psyphy.data import ResponseData

# Model
model = WPPM(dim=2, task=OddityTask(afc=3), prior=Prior.default()) # additional arguments ommited for clarity

# Inference (provides posterior.sample())
inference = LangevinSampler(steps=800, step_size=0.005, temperature=0.001)

# Candidate and eval pools (could be the same; eval can be denser)
stim_bank = make_candidate_pairs(...)   # list[(ref, probe)]
eval_bank = make_eval_pairs(...)        # list[(ref, probe)] or [(ref, ref)] if defined that way

placement = InfoGainPlacement(candidate_pool=stim_bank, eval_pool=eval_bank, threshold=2/3, n_post=128, n_mc=256)

sess = ExperimentSession(model, inference, placement)
data = ResponseData.empty()
posterior = sess.initialize(data)

num_baches = 10

for _ in range(num_batches):  
    batch = sess.next_batch(posterior, batch_size=40)
    present(batch)                                  # experiment computer
    responses = collect_subject_data()
    data = data.append_batch(responses, batch)
    posterior = sess.update(data)                   # full refit/update here

# Readouts
ellipse = posterior.predict_thresholds(reference=[0.0, 0.0], criterion=2/3, directions=24)
save_posterior(posterior, "fits/subjectX.filetype")

```