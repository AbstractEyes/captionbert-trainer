import copy
import pytest
from tools.analyze_structured_final import choose_deployment


def fixture():
    lock = {"analysis": {"status": "selected", "hardware_repeat_floor": .0001,
            "selection": {"structured": "H16", "relative": "N16"}}}
    results = {}
    for stage, arm in (("full4m", "H16"), ("full4m", "N16"), ("exposure1m", "H16")):
        for seed in (1, 2):
            results[f"{stage}/{arm}-seed{seed}"] = dict(
                status="complete", steps=7808,
                training_rows=4000000 if stage == "full4m" else 1000000,
                arm=arm, seed=seed, config_id=stage, dataset_id=stage,
                development=dict(split="validation", spearman=(.72 if arm == "H16" else .70) + seed*.001),
                evaluation=dict(heldout=dict(cos_to_target=.78, erank=80)),
                model_config=dict(rotary_mode="hierarchical" if arm == "H16" else "ntk"),
                parameters=54126848 if arm == "H16" else 54114560, seconds_per_step=.5,
            )
    return lock, results


def test_deployment_uses_pair_development_then_development_seed():
    lock, results = fixture()
    chosen = choose_deployment(lock, results)
    assert (chosen["stage"], chosen["arm"], chosen["seed"]) == ("full4m", "H16", 2)
    assert chosen["test_scores_used"] is False
    assert chosen["full_stage_repeat_measured"] is False


def test_test_values_cannot_change_the_default():
    lock, results = fixture()
    expected = choose_deployment(lock, results)
    for i, item in enumerate(results.values()):
        item["benchmark"] = {"mean": (-100 if i % 2 else 100)}
        item["test_evaluation"] = object()
    assert choose_deployment(lock, results) == expected


def test_near_tie_prefers_simple_reference_but_keeps_raw_leader():
    lock, results = fixture()
    lock["analysis"]["hardware_repeat_floor"] = .03
    chosen = choose_deployment(lock, results)
    assert (chosen["arm"], chosen["raw_mean_leader"]) == ("N16", "H16")


def test_geometry_viability_applies_before_deployment():
    lock, results = fixture()
    for seed in (1, 2):
        results[f"full4m/H16-seed{seed}"]["evaluation"]["heldout"]["erank"] = 20
    chosen = choose_deployment(lock, results)
    assert chosen["arm"] == "N16" and not chosen["structured_viable"]


def test_incomplete_or_mixed_final_artifacts_are_rejected():
    lock, results = fixture()
    missing = copy.deepcopy(results)
    del missing["exposure1m/H16-seed2"]
    with pytest.raises(ValueError, match="six"):
        choose_deployment(lock, missing)
    mixed = copy.deepcopy(results)
    mixed["full4m/N16-seed2"]["config_id"] = "other"
    with pytest.raises(ValueError, match="identity"):
        choose_deployment(lock, mixed)
