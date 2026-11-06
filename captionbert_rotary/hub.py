"""Immutable background uploads with durable retry queues and remote resume."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import time
from .data import atomic_json


class Uploader:
    def __init__(self, cfg, api=None, sleep=time.sleep):
        self.cfg, self.sleep = cfg, sleep
        self.root = Path(cfg.output_dir)
        self.queue = self.root / ".upload_queue"
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.jobs = []
        self.last_checkpoint = {}
        if cfg.push:
            from huggingface_hub import HfApi
            self.api = api or HfApi()
            self.api.create_repo(cfg.hub_repo, repo_type="model", exist_ok=True)
            self.queue.mkdir(parents=True, exist_ok=True)
            for entry in sorted(self.queue.iterdir()):
                if (entry / "job.json").exists():
                    self.jobs.append(self.executor.submit(self._upload, entry))

    def _upload(self, entry):
        job = json.loads((entry / "job.json").read_text())
        for attempt in range(8):
            try:
                if job.get("kind") == "checkpoint" and (self.root / job["name"] / "result.json").exists():
                    self._remove_entry(entry)
                    return True  # The complete snapshot will supersede this resume state.
                print(f"Upload {job['name']}: attempt {attempt + 1}/8 -> "
                      f"https://huggingface.co/{self.cfg.hub_repo}/tree/main/{job['remote']}", flush=True)
                self.api.upload_folder(repo_id=self.cfg.hub_repo, repo_type="model",
                                       folder_path=entry / "payload", path_in_repo=job["remote"],
                                       commit_message=f"Ship {job['name']} (all arms retained)")
                # Delete only a successful spool snapshot owned by this uploader.
                self._remove_entry(entry)
                print(f"Uploaded {job['name']}", flush=True)
                return True
            except Exception as error:
                if attempt == 7:
                    print(f"Upload {job['name']} failed after 8 attempts ({type(error).__name__}); "
                          f"retry snapshot retained at {entry}", flush=True)
                    return False
                delay = min(15 * 2 ** attempt, 600)
                print(f"Upload {job['name']}: {type(error).__name__}; retrying in {delay}s", flush=True)
                self.sleep(delay)

    def submit(self, name):
        if not self.cfg.push:
            return
        # Shipping never stalls the next arm; immutable failed snapshots stay queued.
        self.flush(wait=False)
        entry = self.queue / f"{name}-{time.time_ns()}"
        print(f"Upload {name}: preparing immutable local snapshot...", flush=True)
        shutil.copytree(self.root / name, entry / "payload")
        for manifest in ("experiment.json", "predictions.json", "summary.json", "environment.json",
                         "benchmark_revisions.json"):
            if (self.root / manifest).exists():
                shutil.copy2(self.root / manifest, entry / "payload" / manifest)
        atomic_json(entry / "job.json", {"name": name,
                    "kind": "complete",
                    "remote": f"experiments/{self.cfg.fingerprint()}/{name}"})
        self.jobs.append(self.executor.submit(self._upload, entry))
        print(f"Upload {name}: queued in background", flush=True)

    def _remove_entry(self, entry):
        if entry.resolve().parent != self.queue.resolve():
            raise RuntimeError("Invalid upload queue path")
        shutil.rmtree(entry)

    def submit_analysis(self, name, folder, remote):
        if not self.cfg.push:
            return
        self.flush(wait=False)
        entry = self.queue / f"{name}-analysis-{time.time_ns()}"
        shutil.copytree(folder, entry / "payload")
        atomic_json(entry / "job.json", {"name": name, "kind": "analysis", "remote": remote})
        self.jobs.append(self.executor.submit(self._upload, entry))

    def submit_checkpoint(self, name, step, identity, force=False):
        if not self.cfg.push or step <= 0:
            return
        now = time.monotonic()
        if not force and name in self.last_checkpoint and now - self.last_checkpoint[name] < self.cfg.resume_upload_minutes * 60:
            return
        if any(not future.done() for future in self.jobs):
            print(f"Resume snapshot {name}: prior upload still active; keeping local checkpoint without blocking training", flush=True)
            return
        self.flush()
        source = self.root / name
        entry = self.queue / f"{name}-resume-{time.time_ns()}"
        payload = entry / "payload"
        payload.mkdir(parents=True)
        for path in source.iterdir():
            if path.is_file() and (path.name in ("state.pt", "train.jsonl") or path.name.startswith("eval-")):
                shutil.copy2(path, payload / path.name)
        atomic_json(payload / "resume.json", {**identity, "step": step})
        atomic_json(entry / "job.json", {"name": name, "kind": "checkpoint", "step": step,
                    "remote": f"experiments/{self.cfg.fingerprint()}/{name}"})
        # A newer immutable resume snapshot replaces older failed resume spools for this run.
        for older in list(self.queue.iterdir()):
            if older == entry or not (older / "job.json").exists():
                continue
            job = json.loads((older / "job.json").read_text())
            if job.get("name") == name and job.get("kind") == "checkpoint":
                self._remove_entry(older)
        self.last_checkpoint[name] = now
        self.jobs.append(self.executor.submit(self._upload, entry))
        print(f"Resume snapshot {name}: step {step} queued; interval {self.cfg.resume_upload_minutes:g} min", flush=True)

    def flush(self, wait=True):
        jobs, self.jobs = self.jobs, []
        if jobs and wait:
            print(f"Waiting for {len(jobs)} queued upload(s)...", flush=True)
        for future in jobs:
            if not wait and not future.done():
                self.jobs.append(future)
                continue
            future.result()  # Upload failures return False and retain their durable spool.
        if self.cfg.push:
            pending = []
            for entry in list(self.queue.iterdir()):
                try:
                    pending.append(json.loads((entry / "job.json").read_text())["name"])
                except FileNotFoundError:
                    pass  # A successful background upload may have just removed its spool.
            atomic_json(self.root / "upload_status.json", {"pending_runs": sorted(set(pending))})

    def retry_pending(self):
        if not self.cfg.push:
            return
        self.flush()
        for entry in sorted(self.queue.iterdir()):
            if (entry / "job.json").exists():
                self.jobs.append(self.executor.submit(self._upload, entry))

    def close(self):
        try:
            self.retry_pending()
            self.flush()
        finally:
            self.executor.shutdown(wait=True)


def pull_completed(cfg, dataset_id=None):
    if not cfg.push:
        return
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError
    prefix = f"experiments/{cfg.fingerprint()}/"
    try:
        api = HfApi()
        revision = api.model_info(cfg.hub_repo).sha
        files = api.list_repo_files(cfg.hub_repo, revision=revision)
    except RepositoryNotFoundError:
        return
    markers = [f for f in files if f.startswith(prefix) and f.endswith(("/result.json", "/resume.json"))
               and len(f[len(prefix):].split("/")) == 2]
    if not markers:
        return
    import torch
    names = {f[len(prefix):].split("/")[0] for f in markers}
    selected = []
    for name in sorted(names):
        marker_name = "result.json" if prefix + name + "/result.json" in files else "resume.json"
        from huggingface_hub import hf_hub_download
        marker = json.loads(Path(hf_hub_download(cfg.hub_repo, prefix + name + "/" + marker_name,
                                              revision=revision)).read_text())
        if marker.get("config_id") != cfg.fingerprint() or (dataset_id is not None and marker.get("dataset_id") != dataset_id):
            raise ValueError(f"Incompatible remote run: {name}")
        destination = Path(cfg.output_dir) / name
        if (destination / "result.json").exists() and (destination / "model.pt").exists():
            continue
        if marker_name == "resume.json" and (destination / "state.pt").exists():
            local = torch.load(destination / "state.pt", map_location="cpu", weights_only=True, mmap=True)
            if local["identity"] != {k: marker[k] for k in ("config_id", "dataset_id", "arm", "seed")}:
                raise ValueError(f"Incompatible local checkpoint: {name}")
            if local["step"] >= marker["step"]:
                print(f"Keeping newer local checkpoint {name}: step {local['step']}", flush=True)
                continue
        selected.append(name)
    if not selected:
        return
    wanted = [f for f in files if any(f.startswith(prefix + name + "/") for name in selected)
              and "/analyses/" not in f]
    remote = snapshot_download(cfg.hub_repo, revision=revision, allow_patterns=wanted)
    for name in selected:
        destination = Path(cfg.output_dir) / name
        source = Path(remote) / prefix / name
        destination.mkdir(parents=True, exist_ok=True)
        # Copy completion/resume markers last; every copied file replaces atomically.
        paths = sorted(source.rglob("*"), key=lambda p: p.name in ("result.json", "resume.json"))
        for path in paths:
            if path.is_file():
                target = destination / path.relative_to(source)
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + ".download.tmp")
                shutil.copy2(path, temporary)
                temporary.replace(target)
        revisions = destination / "benchmark_revisions.json"
        if revisions.exists():
            root_revisions = Path(cfg.output_dir) / "benchmark_revisions.json"
            if root_revisions.exists() and root_revisions.read_bytes() != revisions.read_bytes():
                raise ValueError("Shipped runs have inconsistent benchmark revisions")
            shutil.copy2(revisions, root_revisions)
        print(f"Restored shipped run {name}: {'complete' if (destination / 'result.json').exists() else 'partial checkpoint'}", flush=True)
