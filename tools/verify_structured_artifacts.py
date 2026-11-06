"""Verify the complete structured campaign's durable artifact census.

Only small reports are downloaded. Large weights/state/event files are checked
against revision-pinned Hub metadata; native checkpoint loading is established
by each actual frozen probe. No training, optimizer, or GPU use.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FOLLOWUP = "frozen-followup-59ec95344958"
SCREEN_PROBE_SHA = "6bbf7f7cbaca8ce712cf21ca9cd70c25fabdf2af43b05d44fe652665512e5bb3"
FOLLOWUP_PROBE_SHA = "c778258cabc69e1985c03304af0ebf2e8e4be3ad64aba680d9ccaec4bdd5d03a"
PHASE_FOLDER = "frozen-phase-constant-20261106"
PHASE_PROBE_SHA = "1cf1ba909939b1b9f95f100fc87cfe9c21e7698c689658fb125ac522ca2985b9"


def reject_nonfinite(value):
    raise ValueError("Nonfinite JSON measurement: " + value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--screen", default=str(ROOT / ".cache/structured-results/screen-analysis.json"))
    p.add_argument("--final", default=str(ROOT / ".cache/structured-final-results/final-results.json"))
    p.add_argument("--output", default=str(ROOT / "docs/results/structured_artifact_census.json"))
    p.add_argument("--cache", default=str(ROOT / ".cache/census-hub"))
    args = p.parse_args()
    screen = json.loads(Path(args.screen).read_text(encoding="utf8"))
    final = json.loads(Path(args.final).read_text(encoding="utf8"))
    if len(screen["results"]) != 18 or len(final["results"]) != 6:
        p.error("Finish all18 screen and6 larger models before final artifact verification")
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    repo = "AbstractPhil/captionbert-8192-rotary"
    revision = api.model_info(repo).sha
    entries = {}
    config_ids = {item["config_id"] for item in list(screen["results"].values()) + list(final["results"].values())}
    for prefix in [f"experiments/{x}" for x in sorted(config_ids)] + ["campaigns/structured-20261106"]:
        for item in api.list_repo_tree(repo, path_in_repo=prefix, recursive=True, revision=revision):
            if getattr(item, "blob_id", None):
                lfs = item.lfs
                entries[item.path] = dict(bytes=item.size, git_blob_sha1=item.blob_id,
                    lfs_sha256=lfs.sha256 if lfs else None)
    missing, runs, report_cases = [], {}, 0
    common_rows = set()

    def report(prefix, folder, result):
        nonlocal report_cases
        path = prefix + folder + "/report.json"
        if path not in entries:
            missing.append(path)
            return None
        cached = hf_hub_download(repo, path, revision=revision, cache_dir=args.cache)
        raw = Path(cached).read_bytes()
        value = json.loads(raw, parse_constant=reject_nonfinite)
        expected_probe = PHASE_PROBE_SHA if folder == PHASE_FOLDER else (FOLLOWUP_PROBE_SHA if folder == FOLLOWUP else SCREEN_PROBE_SHA)
        if value["probe_source_sha256"] != expected_probe:
            raise ValueError("Frozen probe source differs from its pinned measurement revision: " + path)
        if any(value.get(k) != result[k] for k in ("arm", "seed", "config_id", "dataset_id")):
            raise ValueError("Frozen report identity differs: " + path)
        if value.get("training_performed") or value.get("selection_use"):
            raise ValueError("Wrong frozen measurement scope: " + path)
        parity = value["independent_checkpoint_auto_parity"]
        if not parity["state_tensors_equal"] or parity["metadata_format"] != "pt":
            raise ValueError("Independent trained state versus safe parity missing: " + path)
        if any(parity["loading_info"].get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError("AutoModel loader mismatch: " + path)
        for case in parity["cases"]:
            if case["max_embedding_delta"] > 2e-6 or case["max_hidden_delta"] > 2e-6:
                raise ValueError("Frozen numerical parity exceeds declared tolerance: " + path)
        report_cases += len(parity["cases"])
        common_rows.add(value["row_ids_sha256"])
        expected = [folder + "/row-ids.npy"] + [folder + "/" + name + "-queries.npz" for name in value["measurements"]]
        expected += [folder + "/" + name + "/dev-sts-predictions.npz"
                     for name in value["sentence_measurements"]]
        if value.get("initialization_control") is not None:
            expected.append(folder + "/initialization-queries.npz")
        if folder in (FOLLOWUP, PHASE_FOLDER) and not any(x.startswith(prefix + folder + "/tensorboard/events.out.tfevents.") for x in entries):
            missing.append(prefix + folder + "/tensorboard/events.out.tfevents.*")
        for name in expected:
            if prefix + name not in entries:
                missing.append(prefix + name)
        if folder in (FOLLOWUP, PHASE_FOLDER):
            destination = ROOT / ".cache" / ("structured-results" if result["config_id"] == screen["config_id"] else "structured-final-results")
            if result["config_id"] == screen["config_id"]:
                destination /= f"{result['arm']}-seed{result['seed']}"
            else:
                stage = "full4m" if result["training_rows"] == 4000000 else "exposure1m"
                destination /= f"{stage}/{result['arm']}-seed{result['seed']}"
            destination /= folder
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "report.json").write_bytes(raw)
        return dict(path=path, file_sha256=hashlib.sha256(raw).hexdigest(),
            probe_source_sha256=value["probe_source_sha256"], row_ids_sha256=value["row_ids_sha256"],
            state_tensors_equal=True, output_cases=len(parity["cases"]),
            max_embedding_delta=max(x["max_embedding_delta"] for x in parity["cases"]),
            max_hidden_delta=max(x["max_hidden_delta"] for x in parity["cases"]),
            measured_conditions=list(value["measurements"]),
            lookup_endpoints={name: {key: metrics["all"][key]
                for key in ("record_mass", "exact_record_top1")}
                for name, metrics in value["measurements"].items()},
            initialization_lookup_endpoints=({key: value["initialization_control"]["measurements"]["all"][key]
                for key in ("record_mass", "exact_record_top1")}
                if value.get("initialization_control") is not None else None),
            reconstructed_initialization=value.get("initialization_control") is not None)

    combined = [(f"screen/{key}", value, False) for key, value in screen["results"].items()]
    combined += [(key, value, True) for key, value in final["results"].items()]
    for key, result, larger in combined:
        name = f"{result['arm']}-seed{result['seed']}"
        prefix = f"experiments/{result['config_id']}/{name}/"
        files = {path[len(prefix):]: metadata for path, metadata in entries.items() if path.startswith(prefix)}
        required = ("model.safetensors", "state.pt", "best.safetensors", "trajectory-mid.safetensors",
            "config.json", "modeling_captionbert.py", "configuration_captionbert.py",
            "tokenizer.json", "tokenizer_config.json", "run.json", "result.json", "train.jsonl",
            "environment.json", "pip_freeze.txt", "experiment.json", "dev-sts-predictions.npz")
        missing.extend(prefix + name for name in required if not files.get(name, {}).get("bytes"))
        if larger and "benchmark_revisions.json" not in files:
            missing.append(prefix + "benchmark_revisions.json")
        if files.get("model.safetensors", {}).get("lfs_sha256") != result["checkpoint_sha256"]:
            raise ValueError("Revision-pinned safe weight SHA256 differs: " + key)
        if not any(x.startswith("tensorboard/events.out.tfevents.") for x in files):
            missing.append(prefix + "tensorboard/events.out.tfevents.*")
        if not any(x.startswith("eval-") and x.endswith(".json") for x in files):
            missing.append(prefix + "eval-*.json")
        folders = [FOLLOWUP] if larger else ["frozen-development"]
        if not larger and result["arm"] == "KALLP":
            folders.extend((FOLLOWUP, PHASE_FOLDER))
        reports = [x for folder in folders if (x := report(prefix, folder, result)) is not None]
        runs[key] = dict(prefix=prefix, arm=result["arm"], seed=result["seed"],
            config_id=result["config_id"], dataset_id=result["dataset_id"], steps=result["steps"],
            training_rows=result["training_rows"], final_model_sha256=result["checkpoint_sha256"],
            development_spearman=result["development"]["spearman"],
            files=files, frozen_reports=reports,
            resume_note="state.pt is authoritative; resume.json may describe an earlier transport snapshot")
    if len(common_rows) != 1:
        raise ValueError("Frozen probes do not share identical sampled caption rows")
    operations = {path: metadata for path, metadata in entries.items()
                  if path.startswith("campaigns/structured-20261106/")}
    provenance_paths = sorted(path for path in operations
        if "/data-provenance/lane-A/" in path and path.endswith("/provenance.json"))
    data_provenance = None
    if not provenance_paths:
        missing.append("campaign prepared-data provenance.json")
    else:
        path = provenance_paths[-1]
        cached = hf_hub_download(repo, path, revision=revision, cache_dir=args.cache)
        raw = Path(cached).read_bytes()
        provenance = json.loads(raw, parse_constant=reject_nonfinite)
        expected_full = {item["dataset_id"] for key, item in final["results"].items() if key.startswith("full4m/")}
        if (provenance["data_identities"]["screen"] != screen["analysis"]["dataset_id"]
                or expected_full != {provenance["data_identities"]["full4m"]}
                or provenance["full4m_training"]["rows"] != 4000000
                or provenance["training_performed"] or provenance["gpu_used"]):
            raise ValueError("Prepared-data provenance differs from the actual training census")
        (ROOT / "docs/results/structured_data_provenance.json").write_bytes(raw)
        data_provenance = dict(path=path, file_sha256=hashlib.sha256(raw).hexdigest(),
            data_identities=provenance["data_identities"], full4m_training=provenance["full4m_training"])
    for lane in ("A", "B"):
        for category in ("lane-", "frozen-final-"):
            if not any(("/operations/" + category + lane + "/") in x and x.endswith("/console.log") for x in operations):
                missing.append("campaign operations console: " + category + lane)
    census = dict(repo=repo, source_revision=revision, status="complete" if not missing else "incomplete",
        trained_models=len(runs), screen_models=18, larger_models=6, frozen_output_cases=report_cases,
        common_frozen_row_ids_sha256=next(iter(common_rows)), runs=runs, operations=operations,
        data_provenance=data_provenance,
        missing=missing, scope="This campaign's24 models; earlier experiments remain at their original paths",
        large_file_verification="Revision-pinned Hub LFS SHA256/size metadata; native state loading in frozen reports",
        training_performed=False, gpu_used=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(census, indent=2, allow_nan=False)+"\n", encoding="utf8", newline="\n")
    print(json.dumps({k: v for k, v in census.items() if k not in ("runs", "operations")}, indent=2))
    if missing:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
