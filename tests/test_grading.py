from captionbert_rotary.data import atomic_json
from captionbert_rotary.train import grade_results


def test_grading_requires_both_paired_seeds_and_scores_long_captions(tmp_path):
    def put(arm, seed, cosine):
        atomic_json(tmp_path / f"{arm}-seed{seed}" / "result.json", {
            "validation": {"heldout": {"cos_to_target": cosine, "erank": 10 + cosine, "self_cos": .1},
                           "above_128_tokens": {"n": 12, "cos_to_target": cosine,
                                                "erank": 10 + cosine, "self_cos": .1}},
            "benchmark": {"mean": cosine}})
    put("A0", 1, .5)
    put("A0", 2, .5)
    put("A1", 1, .6)
    report = grade_results(tmp_path)
    assert not report["comparisons_to_A0"]["A1"]["settled"]
    put("A1", 2, .4)
    report = grade_results(tmp_path)
    assert not report["comparisons_to_A0"]["A1"]["settled"]
    assert not report["comparisons_to_A0"]["A1"]["prediction_supported_both_seeds"]
    put("A1", 2, .6)
    for s in (1, 2):
        put("A2", s, .65)
        put("A3", s, .45)
    report = grade_results(tmp_path)
    assert report["comparisons_to_A0"]["A1"]["prediction_supported_both_seeds"]
    assert report["anchor_ablation"]["anchor_prediction_supported_both_seeds"]
