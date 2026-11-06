"""Development-only selection for the registered structured rotary campaign."""
import math
from statistics import mean


SCREEN_ARMS = ("R8", "N16", "H16", "KEND", "KALL", "KALLP", "KALLP_N0", "KALLP_L8")
SCREEN_NAMES = tuple(f"{arm}-seed{seed}" for arm in (*SCREEN_ARMS, "N16-repeat") for seed in (1, 2))


def _number(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Nonfinite or missing {name}: {value}")
    return float(value)


def pair(arm, results):
    records = [results[f"{arm}-seed{seed}"] for seed in (1, 2)]
    scores, teacher, ranks, seconds = [], [], [], []
    for seed, record in zip((1, 2), records):
        if record.get("arm") != arm or record.get("seed") != seed or record.get("status") != "complete":
            raise ValueError(f"Wrong identity or incomplete result for {arm}-seed{seed}")
        if record.get("steps") != 1952 or record.get("training_rows") != 1000000:
            raise ValueError(f"Unexpected screen exposure for {arm}-seed{seed}")
        if record.get("benchmark") is not None or record.get("test_evaluation") is not None:
            raise ValueError("Screen selection must not consume test evaluations")
        if record["development"].get("split") != "validation":
            raise ValueError("Sentence selection must use the validation split")
        scores.append(_number(record["development"]["spearman"], "development score"))
        heldout = record["evaluation"]["heldout"]
        teacher.append(_number(heldout["cos_to_target"], "teacher cosine"))
        ranks.append(_number(heldout["erank"], "effective rank"))
        seconds.append(_number(record["seconds_per_step"], "step time"))
    identities = {(r["config_id"], r["dataset_id"]) for r in records}
    if len(identities) != 1:
        raise ValueError(f"Unmatched seed identities for {arm}")
    return dict(arm=arm, scores=scores, mean_score=mean(scores), seed_spread=abs(scores[1]-scores[0]),
                teacher=teacher, mean_teacher=mean(teacher), ranks=ranks, mean_rank=mean(ranks),
                mean_step_seconds=mean(seconds), model_config=records[0]["model_config"],
                parameters=records[0]["parameters"])


def repeat_floor(arm, pairs):
    original, repeat = pairs[arm], pairs[f"{arm}-repeat"]
    return max(abs(a-b) for a, b in zip(original["scores"], repeat["scores"]))


def simplicity(item):
    cfg = item["model_config"]
    mode = cfg["rotary_mode"]
    bound = mode in ("key_end", "key_all")
    structural = mode in ("hierarchical", "key_end", "key_all")
    components = (int(mode == "ntk") + int(structural) + int(structural and cfg.get("use_value_addresses", True))
                  + int(bound) + int(bound and cfg.get("key_rho", 2.18) != 0)
                  + int(cfg.get("pointer_weight", 0) > 0))
    return components, item["parameters"], item["mean_step_seconds"], item["arm"]


def noise_bar(candidate, reference, hardware):
    return max(hardware, candidate["seed_spread"], reference["seed_spread"])


def choose_near_tie(items, hardware):
    leader = max(items, key=lambda x: (x["mean_score"], x["arm"]))
    near = [x for x in items if leader["mean_score"] - x["mean_score"] <= noise_bar(x, leader, hardware)]
    return min(near, key=simplicity), leader


def compare(candidate, reference, hardware):
    deltas = [a-b for a, b in zip(candidate["scores"], reference["scores"])]
    delta, bar = mean(deltas), noise_bar(candidate, reference, hardware)
    if delta > bar and all(x > 0 for x in deltas):
        verdict = "positive on both tested seeds beyond the measured bar"
    elif delta < -bar and all(x < 0 for x in deltas):
        verdict = "negative on both tested seeds beyond the measured bar"
    else:
        verdict = "unresolved within repeat/seed variation or with a seed-direction reversal"
    return dict(candidate=candidate["arm"], reference=reference["arm"], seed_deltas=deltas,
                mean_delta=delta, noise_bar=bar, verdict=verdict)


def select_screen(results):
    missing = [name for name in SCREEN_NAMES if name not in results]
    if missing:
        return dict(status="incomplete", missing=missing, selection=None)
    pairs = {arm: pair(arm, results) for arm in (*SCREEN_ARMS, "N16-repeat")}
    identities = {(r["config_id"], r["dataset_id"]) for name, r in results.items() if name in SCREEN_NAMES}
    if len(identities) != 1:
        raise ValueError("Lanes disagree on configuration or data identity")
    hardware = repeat_floor("N16", pairs)
    relative, relative_leader = choose_near_tie([pairs["R8"], pairs["N16"]], hardware)
    if relative["arm"] == "R8":
        additional = [f"R8-repeat-seed{s}" for s in (1, 2) if f"R8-repeat-seed{s}" not in results]
        if additional:
            return dict(status="reference_repeat_required", missing=additional, selection=None,
                        provisional_relative="R8", pairs=pairs, n16_hardware_repeat=hardware)
        pairs["R8-repeat"] = pair("R8-repeat", results)
        for record in (results[f"R8-repeat-seed{s}"] for s in (1, 2)):
            if (record["config_id"], record["dataset_id"]) not in identities:
                raise ValueError("R8 repeat does not match the original screen identity")
        hardware = max(hardware, repeat_floor("R8", pairs))
        relative, relative_leader = choose_near_tie([pairs["R8"], pairs["N16"]], hardware)
    structured = [pairs[a] for a in SCREEN_ARMS if a not in ("R8", "N16")]
    for item in structured:
        item["viable"] = (item["mean_teacher"] >= relative["mean_teacher"] - .02
                          and item["mean_rank"] >= .8 * relative["mean_rank"]
                          and all(a >= .5*b for a, b in zip(item["ranks"], relative["ranks"])))
    viable = [x for x in structured if x["viable"]]
    selected, structured_leader = choose_near_tie(viable or structured, hardware)
    contrasts = [("KALL", "KEND"), ("KALLP", "KALL"), ("KALLP_N0", "KALLP"),
                 ("KALLP_L8", "KALLP"), ("H16", "N16"), (selected["arm"], relative["arm"])]
    return dict(status="selected", config_id=next(iter(identities))[0], dataset_id=next(iter(identities))[1],
                selection=dict(structured=selected["arm"], relative=relative["arm"], seeds=[1, 2],
                               exploratory_rescue=not bool(viable),
                               raw_mean_leaders=dict(structured=structured_leader["arm"], relative=relative_leader["arm"]),
                               rule="mean validation Spearman; within pairwise noise bar prefer fewer components, fewer parameters, then lower measured step time"),
                hardware_repeat_floor=hardware, pairs=pairs,
                contrasts=[compare(pairs[a], pairs[b], hardware) for a, b in contrasts],
                claim_limit="Descriptive paired-seed evidence in this recipe; no population significance or long-context semantic claim")
