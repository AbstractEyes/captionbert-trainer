"""Parent-process output relay for notebook cells and an on-disk console log."""


def run_logged(command, log_path, env=None):
    import os
    from pathlib import Path
    import subprocess
    import time

    child_env = os.environ.copy() if env is None else dict(env)
    child_env["PYTHONUNBUFFERED"] = "1"
    child_env["PYTHONIOENCODING"] = "utf-8"
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Streaming child output; console log: {path}", flush=True)
    started = time.monotonic()
    with path.open("a", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1,
                                   env=child_env)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            returncode = process.wait()
        except KeyboardInterrupt:
            import signal
            process.send_signal(signal.SIGINT)
            raise
        finally:
            process.stdout.close()
        print(f"Child exited with code {returncode}; elapsed {time.monotonic() - started:.1f}s; "
              f"console log: {path}", flush=True)
        if returncode:
            raise subprocess.CalledProcessError(returncode, command)
    return returncode
