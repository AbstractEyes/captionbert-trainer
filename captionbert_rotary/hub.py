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
        for attempt in range(5):
            try:
                self.api.upload_folder(repo_id=self.cfg.hub_repo, repo_type="model",
                                       folder_path=entry / "payload", path_in_repo=job["remote"],
                                       commit_message=f"Ship {job['name']} (all arms retained)")
                # Delete only a successful spool snapshot owned by this uploader.
                if entry.resolve().parent != self.queue.resolve():
                    raise RuntimeError("Invalid upload queue path")
                shutil.rmtree(entry)
                print(f"Uploaded {job['name']}", flush=True)
                return
            except Exception:
                if attempt == 4:
                    raise
                self.sleep(min(2 ** attempt, 30))

    def submit(self, name):
        if not self.cfg.push:
            return
        # Bound outstanding work to one arm: a lost runtime can lose at most the active arm.
        self.flush()
        entry = self.queue / f"{name}-{time.time_ns()}"
        shutil.copytree(self.root / name, entry / "payload")
        for manifest in ("experiment.json", "predictions.json", "summary.json", "environment.json",
                         "benchmark_revisions.json"):
            if (self.root / manifest).exists():
                shutil.copy2(self.root / manifest, entry / "payload" / manifest)
        atomic_json(entry / "job.json", {"name": name,
                    "remote": f"experiments/{self.cfg.fingerprint()}/{name}"})
        self.jobs.append(self.executor.submit(self._upload, entry))

    def flush(self):
        jobs, self.jobs = self.jobs, []
        for future in jobs:
            future.result()  # Failed pushes remain local and stop the next arm.

    def close(self):
        try:
            self.flush()
        finally:
            self.executor.shutdown(wait=True)


def pull_completed(cfg):
    if not cfg.push:
        return
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError
    prefix = f"experiments/{cfg.fingerprint()}/"
    try:
        files = HfApi().list_repo_files(cfg.hub_repo)
    except RepositoryNotFoundError:
        return
    complete = [f for f in files if f.startswith(prefix) and f.endswith("/result.json")]
    if not complete:
        return
    names = {f[len(prefix):].split("/")[0] for f in complete}
    remote = snapshot_download(cfg.hub_repo, allow_patterns=[prefix + name + "/*" for name in names])
    for name in names:
        destination = Path(cfg.output_dir) / name
        if (destination / "result.json").exists():
            continue
        shutil.copytree(Path(remote) / prefix / name, destination, dirs_exist_ok=True)
        revisions = destination / "benchmark_revisions.json"
        if revisions.exists():
            root_revisions = Path(cfg.output_dir) / "benchmark_revisions.json"
            if root_revisions.exists() and root_revisions.read_bytes() != revisions.read_bytes():
                raise ValueError("Shipped runs have inconsistent benchmark revisions")
            shutil.copy2(revisions, root_revisions)
        print(f"Resumed shipped arm {name}", flush=True)
