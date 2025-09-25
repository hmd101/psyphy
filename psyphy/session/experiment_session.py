"""
session/experiment_session.py
-----------------------------

ExperimentSession orchestrates the data -> model -> inference -> posterior -> trial placement loop.

Responsibilities:
- Initialize posterior from prior and/or data
- Update posterior after new responses
- Query TrialPlacement for next batch
- Manage serialization of session state

End-to-end entry point for running adaptive experiments.
"""

from psyphy.data.dataset import ResponseData


class ExperimentSession:
    def __init__(self, model, inference, placement, init_placement=None):
        self.model = model
        self.inference = inference
        self.placement = placement
        self.init_placement = init_placement
        self.data = ResponseData()
        self.posterior = None

    def initialize(self):
        # Fit once before any adaptive placement
        self.posterior = self.inference.fit(self.model, self.data)
        return self.posterior

    def update(self):
        # Re-fit posterior with accumulated data
        self.posterior = self.inference.fit(self.model, self.data)
        return self.posterior

    def next_batch(self, batch_size: int):
        return self.placement.propose(self.posterior, batch_size)
