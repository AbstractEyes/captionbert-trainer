from dataclasses import asdict
import hashlib
from pathlib import Path
from types import SimpleNamespace
import torch
from captionbert_rotary import build_model
from captionbert_rotary.data import atomic_json
from captionbert_rotary.diagnostics import PROTOCOL
from captionbert_rotary.model_card import TASK_NAMES, render_model_card, finalize_model_card


def complete_screen():
    results, reports, hashes = {}, {}, {}
    config = {"arms": ["A0", "A1", "A2", "A3"], "seeds": [1, 2]}
    for arm in config["arms"]:
        for seed in config["seeds"]:
            key = arm, seed
            hashes[key] = hashlib.sha256(f"{arm}-{seed}".encode()).hexdigest()
            results[key] = {"status": "complete", "config_id": "experiment", "arm": arm,
                "seed": seed, "steps": 1952, "dataset_id": "data", "parameters": 54114560,
                "validation": {s: {"cos_to_target": .7, "erank": 60., "self_cos": .02, "n": 5000}
                               for s in ("heldout", "above_128_tokens")},
                "benchmark": {"mean": .5, "tasks": {t: {"spearman": .5} for t in TASK_NAMES},
                              "dataset_revisions": {"test-dataset": "revision"}, "max_tokens": 64, "split": "test"}}
            reports[key] = {"protocol": PROTOCOL, "status": "measured", "baseline_finite": True,
                "identity": {"arm": arm, "seed": seed, "config_id": "experiment", "dataset_id": "data",
                             "step": 1952, "checkpoint_sha256": hashes[key]},
                "probe_sha256": "same-probe", "repeat_floor": {"max_absolute_change": 0.},
                "interventions": {c: {"mean_l2_change": .1 if arm == "A2" else 0.}
                                  for c in ("sentence_ids_collapsed", "remove_anchor")}}
    return results, reports, config, hashes


def render(results, reports, config, hashes):
    return render_model_card(results, reports, config, "test/repo", "experiment", "pinned-revision", hashes)


def test_card_uses_v2_format_actual_scores_and_pinned_package_loading():
    results, reports, config, hashes = complete_screen()
    text, status = render(results, reports, config, hashes)
    assert status["ready"] and not status["missing_runs"] and not status["problems"]
    assert "## Benchmark" in text and "## Geometry" in text and "## Output convention" in text
    assert 'revision="pinned-revision"' in text and "SentenceEncoder.from_pretrained" in text
    assert "| A2 / 2 |" in text and "0.5000" in text
    assert "short screen" in text and "not a root-level Transformers" in text
    assert "captionbert-8192-v2/blob/ed468fe5" in status["format_reference"]


def test_incomplete_or_invalid_evidence_cannot_make_a_final_card():
    results, reports, config, hashes = complete_screen()
    results.pop(("A3", 2))
    text, status = render(results, reports, config, hashes)
    assert not status["ready"] and status["missing_runs"] == ["A3-seed2"]
    assert "Incomplete draft" in text
    results, reports, config, hashes = complete_screen()
    reports["A2", 1]["identity"]["checkpoint_sha256"] = "wrong-weights"
    results["A1", 2]["benchmark"]["tasks"]["STS-B"]["spearman"] = "unmeasured"
    results["A0", 2]["benchmark"]["dataset_revisions"] = {"different": "revision"}
    _, status = render(results, reports, config, hashes)
    assert not status["ready"]
    assert any("mismatched" in p for p in status["problems"])
    assert any("revisions differ" in p for p in status["problems"])


def test_finalize_publishes_only_complete_evidence_with_concurrency_guard(tmp_path, monkeypatch):
    import huggingface_hub
    results, reports, config, hashes = complete_screen()
    remote = tmp_path / "remote"
    prefix = "experiments/experiment"
    for key, result in results.items():
        run = remote / prefix / f"{key[0]}-seed{key[1]}"
        atomic_json(run / "result.json", result)
        atomic_json(run / "experiment.json", {"config": config})
        atomic_json(run / f"analyses/{PROTOCOL}/rotary_diagnostics.json", reports[key])
        (run / "model.pt").write_bytes(b"metadata-only-test")
    calls = []
    class API:
        def model_info(self, *args, **kwargs):
            assert kwargs["files_metadata"]
            assets = []
            for p in remote.rglob("*"):
                if not p.is_file():
                    continue
                name = p.relative_to(remote).as_posix()
                run = p.parent.name
                lfs = None
                if p.name == "model.pt":
                    arm, seed = run.split("-seed")
                    lfs = {"sha256": hashes[arm, int(seed)]}
                assets.append(SimpleNamespace(rfilename=name, lfs=lfs))
            return SimpleNamespace(sha="source-revision", siblings=assets)
        def upload_file(self, **kwargs):
            assert kwargs["parent_commit"] == "source-revision" and kwargs["path_in_repo"] == "README.md"
            calls.append(Path(kwargs["path_or_fileobj"]).read_text())
            return SimpleNamespace(oid="published-revision")
    monkeypatch.setattr(huggingface_hub, "HfApi", API)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, filename, **kw: str(remote / filename))
    one_report = remote / prefix / f"A2-seed1/analyses/{PROTOCOL}/rotary_diagnostics.json"
    saved = one_report.read_bytes()
    one_report.unlink()
    draft = finalize_model_card("test/repo", "experiment", tmp_path / "draft", push=True)
    assert not draft["published"] and not calls
    one_report.write_bytes(saved)
    done = finalize_model_card("test/repo", "experiment", tmp_path / "final", push=True)
    assert done["published"] and done["published_revision"] == "published-revision" and len(calls) == 1


def test_hub_inference_fetches_only_final_model_result_and_tokenizer(tmp_path, tiny_config, monkeypatch):
    import huggingface_hub
    import transformers
    from captionbert_rotary.inference import SentenceEncoder
    run = "experiments/id/A2-seed1"
    source = tmp_path / run
    atomic_json(source / "result.json", {"arm": "A2", "model_config": asdict(tiny_config)})
    torch.save(build_model("A2", tiny_config).state_dict(), source / "model.pt")
    calls = []
    def snapshot(repo, **kwargs):
        calls.append(kwargs)
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: object())
    encoder = SentenceEncoder.from_pretrained("test/repo", run=run, revision="chosen-revision")
    assert encoder.device == "cpu"
    assert calls == [{"revision": "chosen-revision", "allow_patterns": [
        run + "/result.json", run + "/model.pt", run + "/tokenizer/*"]}]
