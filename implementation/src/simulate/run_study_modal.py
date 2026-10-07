"""Run the case study on Modal: every run of run_study.sh at once, split by population size.

The runs are the ones `run_study.sh --list` prints, so the laptop and Modal can never run
different studies. Each run is split into one container per household count in
config/study.yaml, since every count is compared on its own. Each piece is kept in the Modal
volume tcc-case-study as it finishes, and the pieces of a run are merged into the same
results/study/study_<tag>_seed<n>.{csv,json,txt} that run_study.sh writes.

Launch detached, so the study keeps running if this machine sleeps or disconnects, and
collect whatever has finished at any time:
    modal run --detach src/simulate/run_study_modal.py
    modal run src/simulate/run_study_modal.py --collect
    modal run --detach src/simulate/run_study_modal.py --only main --label smoke \\
        --extra "--test-days 1 --calibration-days 1" --out /tmp/smoke   # a short check
Then: python src/analyze/summarize_study.py -> results/study/SUMMARY.md
"""

import json
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
"""Where the merged runs are written."""

VOLUME = "/results"
"""Where the volume is mounted in a container."""

SUFFIXES = (".csv", ".json", ".txt")
"""The files every run writes."""

image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy==2.5.2", "pyyaml==6.0.3")
for shipped in ("src", "config", "data", "results/measurements"):
    image = image.add_local_dir(IMPLEMENTATION / shipped, str(REMOTE / shipped))

results_volume = modal.Volume.from_name("tcc-case-study", create_if_missing=True)

app = modal.App("tcc-case-study")


# ==========================================
# Merging the pieces of a run
# ==========================================
# Pure functions, so the merge can be checked without Modal.
def piece_name(tag: str, seed: int, subscribers: int) -> str:
    """Return the file stem of one population's piece of a run."""
    return f"study_{tag}_seed{seed}_n{subscribers}"


def merge_pieces(pieces: list[dict]) -> dict:
    """Merge one run's pieces, one per household count, into the files simulate.py writes.

    Every household count is compared on its own (rows, frontiers, savings), so a run split
    by count and merged equals the run made whole.

    Args:
        pieces: {suffix: bytes} per piece, in household-count order.

    Returns:
        {suffix: bytes} of the merged run.
    """
    reports = [json.loads(piece[".json"]) for piece in pieces]
    merged = reports[0]
    for key in ("configs", "rows", "frontiers", "hourly"):
        merged[key] = [entry for report in reports for entry in report[key]]
    merged["args"]["subscribers"] = ",".join(
        str(config["subscribers"]) for config in merged["configs"]
    )
    tables = [piece[".csv"].splitlines(keepends=True) for piece in pieces]
    csv = b"".join([tables[0][0]] + [line for table in tables for line in table[1:]])
    printout = b"\n".join(piece[".txt"] for piece in pieces)
    return {
        ".csv": csv,
        ".json": json.dumps(merged, indent=1).encode(),
        ".txt": printout,
    }


# ==========================================
# One piece, in a container
# ==========================================
@app.function(
    image=image,
    cpu=1.0,
    memory=8192,
    timeout=24 * 60 * 60,
    volumes={VOLUME: results_volume},
)
def simulate_piece(tag: str, seed: int, flags: list[str], subscribers: int, label: str) -> dict:
    """Run simulate.py for one household count of one run, and keep what it wrote.

    Args:
        tag: The scenario's name.
        seed: The random seed.
        flags: The settings this scenario changes.
        subscribers: The household count of this piece.
        label: The volume folder the piece is kept in.

    Returns:
        {"name", "returncode", "files": {suffix: bytes}}, bytes so files land as written.
    """
    name = piece_name(tag, seed, subscribers)
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
        "--subscribers",
        str(subscribers),
        "--out",
        str(out),
    ]
    result = subprocess.run(command, cwd=REMOTE, capture_output=True, text=True, check=False)
    files = {".txt": (result.stdout + result.stderr).encode()}
    for suffix in (".csv", ".json"):
        written = out.with_suffix(suffix)
        if written.exists():
            files[suffix] = written.read_bytes()

    # --- Keep it, whether or not the launching machine is still there ---
    if result.returncode == 0:
        folder = Path(VOLUME) / label
        folder.mkdir(parents=True, exist_ok=True)
        for suffix, content in files.items():
            (folder / f"{name}{suffix}").write_bytes(content)
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


def study_sizes() -> list[int]:
    """Return the household counts config/study.yaml sweeps."""
    import yaml  # pylint: disable=import-outside-toplevel

    with open(IMPLEMENTATION / "config" / "study.yaml", encoding="utf-8") as file:
        return [int(size) for size in yaml.safe_load(file)["traffic"]["subscribers"]]


def merge_finished(runs: list[tuple[str, int, list[str]]], sizes: list[int], folder: Path) -> list:
    """Merge every run whose pieces are all in folder/parts/, and return the runs still missing."""
    parts, missing = folder / "parts", []
    for tag, seed, _ in runs:
        # names like study_onu0.5_... hold a dot, so suffixes are appended, never swapped
        stems = [piece_name(tag, seed, size) for size in sizes]
        if not all((parts / f"{stem}.json").exists() for stem in stems):
            missing.append(f"study_{tag}_seed{seed}")
            continue
        pieces = [
            {suffix: (parts / f"{stem}{suffix}").read_bytes() for suffix in SUFFIXES}
            for stem in stems
        ]
        for suffix, content in merge_pieces(pieces).items():
            (folder / f"study_{tag}_seed{seed}{suffix}").write_bytes(content)
    return missing


@app.local_entrypoint()
def main(
    only: str = "", extra: str = "", out: str = "", label: str = "study", collect: bool = False
):
    """Launch every piece not yet collected, or collect what the volume holds, then merge.

    Args:
        only: Comma-separated scenario tags (all when empty).
        extra: Flags added to every run, for a short check (e.g. "--test-days 1").
        out: Folder to write to instead of results/study/.
        label: The volume folder pieces are kept in; use another for a check.
        collect: Download the finished pieces from the volume instead of launching.
    """
    folder = Path(out) if out else STUDY
    parts = folder / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    wanted = {tag for tag in only.split(",") if tag}
    runs = [
        (tag, seed, flags + extra.split())
        for tag, seed, flags in study_jobs()
        if not wanted or tag in wanted
    ]
    sizes = study_sizes()

    if collect:
        # --- Bring down whatever finished, even while this machine was away ---
        names = [entry.path.split("/")[-1] for entry in results_volume.listdir(label)]
        for name in names:
            if not (parts / name).exists():
                (parts / name).write_bytes(b"".join(results_volume.read_file(f"{label}/{name}")))
        print(f"{len(names)} files in the volume's {label}/")
    else:
        # --- Launch every piece not yet here ---
        jobs = [
            (tag, seed, flags, size, label)
            for tag, seed, flags in runs
            for size in sizes
            if not (parts / f"{piece_name(tag, seed, size)}.json").exists()
        ]
        print(f"{len(jobs)} pieces on Modal, one container each, writing to {folder}")
        for result in simulate_piece.starmap(jobs, order_outputs=False, return_exceptions=True):
            if isinstance(result, Exception):
                print(f"  failed: {result!r}", flush=True)
                continue
            for suffix, content in result["files"].items():
                (parts / f"{result['name']}{suffix}").write_bytes(content)
            status = "failed (see its .txt)" if result["returncode"] else "done"
            print(f"  {status}: {result['name']}", flush=True)

    missing = merge_finished(runs, sizes, folder)
    print(f"{len(runs) - len(missing)} of {len(runs)} runs merged into {folder}")
    if missing:
        print("still missing pieces: " + ", ".join(missing))
