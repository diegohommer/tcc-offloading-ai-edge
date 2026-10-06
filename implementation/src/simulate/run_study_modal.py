"""Run the case study on Modal: every run of run_study.sh at once, one CPU container each.

The runs are the ones `run_study.sh --list` prints, so the laptop and Modal can never run
different studies. Each container runs simulate.py exactly as run_study.sh would, and its
CSV, JSON and printout land in results/study/ as soon as it finishes. Runs already there
are skipped, so the study can be restarted. Each container also keeps its files in the
Modal volume tcc-case-study, so a run finished while the launching machine was away is not
lost: `modal volume get tcc-case-study / results/study/` fetches them.

Usage:
    modal run src/simulate/run_study_modal.py
    modal run src/simulate/run_study_modal.py --only main,burst_all   # a subset of tags
    modal run src/simulate/run_study_modal.py --only main \
        --extra "--test-days 1 --calibration-days 1 --subscribers 1000" --out /tmp/smoke
Then: python src/analyze/summarize_study.py -> results/study/SUMMARY.md
"""

import subprocess
import sys
from pathlib import Path

import modal

# ==========================================
# Settings
# ==========================================
REMOTE = Path("/root/implementation")
"""Where implementation/ sits inside the container."""

IMPLEMENTATION = Path(__file__).resolve().parents[2] if modal.is_local() else REMOTE
"""The implementation/ folder: the checkout on the machine that launches, REMOTE in a container."""

STUDY = IMPLEMENTATION / "results" / "study"
"""Where the runs are written."""

image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.5.2", "pyyaml==6.0.3")
for shipped in ("src", "config", "data", "results/measurements"):
    image = image.add_local_dir(IMPLEMENTATION / shipped, str(REMOTE / shipped))

VOLUME = "/results"
"""Where the volume is mounted in a container."""

results_volume = modal.Volume.from_name("tcc-case-study", create_if_missing=True)

app = modal.App("tcc-case-study")


# ==========================================
# One run, in a container
# ==========================================
@app.function(
    image=image,
    cpu=1.0,
    memory=8192,
    timeout=10 * 60 * 60,
    volumes={VOLUME: results_volume},
)
def simulate_run(tag: str, seed: int, flags: list[str]) -> dict:
    """Run simulate.py for one run of the study, and return what it wrote.

    Args:
        tag: The scenario's name.
        seed: The random seed.
        flags: The settings this scenario changes.

    Returns:
        {"name", "returncode", "files": {suffix: bytes}} for the run's CSV, JSON and printout,
        as bytes so the files land exactly as written.
    """
    name = f"study_{tag}_seed{seed}"
    out = Path("/tmp/study") / f"{name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "src/simulate/simulate.py",
        "--config",
        "config/study.yaml",
        "--seed",
        str(seed),
        *flags,
        "--out",
        str(out),
    ]
    result = subprocess.run(command, cwd=REMOTE, capture_output=True, text=True, check=False)
    files = {".txt": (result.stdout + result.stderr).encode()}
    for suffix in (".csv", ".json"):
        written = out.with_suffix(suffix)
        if written.exists():
            files[suffix] = written.read_bytes()

    # --- Keep a copy, in case the launching machine is gone when the run finishes ---
    if result.returncode == 0:
        for suffix, content in files.items():
            (Path(VOLUME) / f"{name}{suffix}").write_bytes(content)
        results_volume.commit()
    return {"name": name, "returncode": result.returncode, "files": files}


# ==========================================
# The study, from the launching machine
# ==========================================
def study_jobs() -> list[tuple[str, int, list[str]]]:
    """Return the study's runs, as run_study.sh lists them: (tag, seed, flags)."""
    listing = subprocess.run(
        ["bash", str(IMPLEMENTATION / "src" / "simulate" / "run_study.sh"), "--list"],
        capture_output=True,
        text=True,
        check=True,
    )
    jobs = []
    for line in listing.stdout.splitlines():
        tag, seed, *flags = line.split()
        jobs.append((tag, int(seed), flags))
    return jobs


@app.local_entrypoint()
def main(only: str = "", extra: str = "", out: str = ""):
    """Launch every run not yet written, and write each one as it finishes.

    Args:
        only: Comma-separated scenario tags to run (all when empty).
        extra: Flags added to every run, for a short check (e.g. "--test-days 1").
        out: Folder to write to instead of results/study/.
    """
    folder = Path(out) if out else STUDY
    wanted = {tag for tag in only.split(",") if tag}
    jobs = [
        (tag, seed, flags + extra.split())
        for tag, seed, flags in study_jobs()
        if (not wanted or tag in wanted) and not (folder / f"study_{tag}_seed{seed}.json").exists()
    ]
    if not jobs:
        print(f"every run is already in {folder}")
        return
    print(f"{len(jobs)} runs on Modal, one container each, writing to {folder}")

    folder.mkdir(parents=True, exist_ok=True)
    failed = []
    for result in simulate_run.starmap(jobs, order_outputs=False, return_exceptions=True):
        if isinstance(result, Exception):
            failed.append(repr(result))
            print(f"  failed: {result!r}", flush=True)
            continue
        for suffix, content in result["files"].items():
            (folder / f"{result['name']}{suffix}").write_bytes(content)
        if result["returncode"]:
            failed.append(result["name"])
            print(f"  failed: {result['name']} (see its .txt)", flush=True)
        else:
            print(f"  done: {result['name']}", flush=True)

    print(f"\n{len(jobs) - len(failed)} of {len(jobs)} runs written to {folder}")
    if failed:
        print("failed: " + ", ".join(failed))
