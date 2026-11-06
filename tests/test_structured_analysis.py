import copy
import pytest
from captionbert_rotary.structured_analysis import SCREEN_NAMES, select_screen


def results():
    values = {}
    for name in SCREEN_NAMES:
        arm, seed = name.rsplit("-seed", 1)
        key = arm not in ("R8", "N16", "N16-repeat", "H16")
        score = .6 if arm == "R8" else (.72 if arm == "KALLP" else .65)
        values[name] = dict(arm=arm, seed=int(seed), status="complete", steps=1952,
            training_rows=1000000, config_id="same", dataset_id="same-data",
            benchmark=None, test_evaluation=None, development=dict(split="validation", spearman=score),
            evaluation=dict(heldout=dict(cos_to_target=.8, erank=50)), seconds_per_step=.5, parameters=100,
            model_config=dict(rotary_mode="key_all" if key else ("hierarchical" if arm=="H16" else ("rope" if arm=="R8" else "ntk")),
                              pointer_weight=.1 if arm.startswith("KALLP") else 0, key_rho=2.18))
    return values


def test_requires_complete_pairs_and_preserves_an_incomplete_status():
    value=results()
    del value["KALLP-seed2"]
    report=select_screen(value)
    assert report["selection"] is None and report["missing"]==["KALLP-seed2"]


def test_development_only_promotion_and_relevant_noise():
    value=results()
    report=select_screen(value)
    assert report["selection"]["structured"]=="KALLP"
    assert report["selection"]["relative"]=="N16"
    assert not report["selection"]["exploratory_rescue"]
    value["KALLP-seed2"]["evaluation"]["heldout"]["erank"]=10
    report=select_screen(value)
    assert not report["pairs"]["KALLP"]["viable"]
    assert report["selection"]["structured"]!="KALLP"


@pytest.mark.parametrize("fault", ["test", "identity", "nan"])
def test_refuses_selection_leakage_or_mismatched_artifacts(fault):
    value=results()
    if fault=="test": value["R8-seed1"]["benchmark"]={"test":.9}
    elif fault=="identity": value["KALLP-seed2"]["dataset_id"]="other"
    else: value["KALLP-seed2"]["development"]["spearman"]=float("nan")
    with pytest.raises(ValueError): select_screen(value)


def test_requires_a_repeat_of_the_selected_eight_head_reference():
    value=results()
    for seed in (1,2): value[f"R8-seed{seed}"]["development"]["spearman"]=.8
    report=select_screen(value)
    assert report["status"]=="reference_repeat_required"
    for seed in (1,2):
        item=copy.deepcopy(value[f"R8-seed{seed}"])
        item["arm"]="R8-repeat"
        value[f"R8-repeat-seed{seed}"]=item
    report=select_screen(value)
    assert report["selection"]["relative"]=="R8"


def test_a_seed_direction_reversal_does_not_become_a_positive_claim():
    value=results()
    value["KALLP-seed1"]["development"]["spearman"]=.80
    value["KALLP-seed2"]["development"]["spearman"]=.60
    report=select_screen(value)
    contrast=next(x for x in report["contrasts"] if x["candidate"]=="KALLP" and x["reference"]=="KALL")
    assert contrast["verdict"].startswith("unresolved")
    assert contrast["seed_deltas"][0]>0 and contrast["seed_deltas"][1]<0
