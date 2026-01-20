def test_top_level_api_imports():
    import psyphy as p

    for name in [
        "WPPM",
        "Prior",
        "OddityTask",
        "GaussianNoise",
        "StudentTNoise",
        "MAPOptimizer",
        "LangevinSampler",
        "LaplaceApproximation",
        "Posterior",
        "ExperimentSession",
        "ResponseData",
        "TrialBatch",
    ]:
        assert hasattr(p, name)
