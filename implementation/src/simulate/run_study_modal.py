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
    modal run --detach src/simulate/run_study_modal.py --self-calibration --only main
        # static_day and static_hour relearned under themselves, split by beta, added to
        # the runs already in results/study/ as static_day_self and static_hour_self
    modal run --detach src/simulate/run_study_modal.py --only main --label smoke \\
        --extra "--test-days 1 --calibration-days 1" --out /tmp/smoke   # a short check
Then: python src/analyze/summarize_study.py -> results/study/SUMMARY.md
"""

import csv
import io
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
    merged_table = b"".join([tables[0][0]] + [line for table in tables for line in table[1:]])
    printout = b"\n".join(piece[".txt"] for piece in pieces)
    return {
        ".csv": merged_table,
        ".json": json.dumps(merged, indent=1).encode(),
        ".txt": printout,
    }


STATIC = ("static_day", "static_hour")
"""The policies whose tables self-calibration relearns; added back as <policy>_self."""


def beta_part(beta: float) -> str:
    """Return the name suffix of one beta's piece."""
    return f"_b{beta:g}"


def rows_csv(rows: list[dict]) -> bytes:
    """Return result rows as simulate.py writes its CSV."""
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return table.getvalue().encode()


def add_self_calibrated(report: dict, new_rows: list[dict]) -> dict:
    """Add a run's static policies relearned under themselves, beside the originals.

    The relearned rows become static_day_self and static_hour_self, next to the tables
    learned under RecServe, so both timetables can be compared. The savings against
    RecServe and every policy's frontier are recomputed per household count. Adding again
    replaces what an earlier addition put there.

    Args:
        report: One run's JSON.
        new_rows: The static policies' rows from runs with --calibration self.

    Returns:
        The report, changed in place.
    """
    sys.path.insert(0, str(IMPLEMENTATION / "src" / "simulate"))
    from frontier import frontier, iso_accuracy  # pylint: disable=import-outside-toplevel

    relearned = {
        (row["subscribers"], row["beta"], row["policy"]): {**row, "policy": f"{row['policy']}_self"}
        for row in new_rows
        if row["policy"] in STATIC
    }
    rows = []
    for row in report["rows"]:
        if row["policy"].endswith("_self"):
            continue
        rows.append(row)
        if (row["subscribers"], row["beta"], row["policy"]) in relearned:
            rows.append(relearned[(row["subscribers"], row["beta"], row["policy"])])
    targets = [float(target) for target in report["args"]["acc_targets"].split(",")]
    policies = list(dict.fromkeys(row["policy"] for row in rows))
    frontiers = []
    for subscribers in dict.fromkeys(row["subscribers"] for row in rows):
        block = [row for row in rows if row["subscribers"] == subscribers]
        iso_accuracy(block)
        frontiers += [
            {
                "subscribers": subscribers,
                "policy": policy,
                "J_at_accuracy": frontier(
                    [row for row in block if row["policy"] == policy], targets
                ),
            }
            for policy in policies
        ]
    report["rows"], report["frontiers"] = rows, frontiers
    listed = report["args"]["policies"].split(",")
    report["args"]["policies"] = ",".join(
        listed + [f"{policy}_self" for policy in STATIC if f"{policy}_self" not in listed]
    )
    return report


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
def simulate_piece(
    tag: str, seed: int, flags: list[str], subscribers: int, label: str, part: str = ""
) -> dict:
    """Run simulate.py for one household count of one run, and keep what it wrote.

    Args:
        tag: The scenario's name.
        seed: The random seed.
        flags: The settings this scenario changes.
        subscribers: The household count of this piece.
        label: The volume folder the piece is kept in.
        part: Appended to the piece's name, for a piece split further (by beta).

    Returns:
        {"name", "returncode", "files": {suffix: bytes}}, bytes so files land as written.
    """
    name = piece_name(tag, seed, subscribers) + part
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


def study_betas() -> list[float]:
    """Return the betas config/study.yaml sweeps."""
    sys.path.insert(0, str(IMPLEMENTATION / "src" / "simulate"))
    from simulate import load_config  # pylint: disable=import-outside-toplevel

    config = load_config(IMPLEMENTATION / "config" / "study.yaml")
    return [float(beta) for beta in config["betas"].split(",")]


def splice_finished(runs: list, sizes: list[int], betas: list[float], folder: Path) -> list:
    """Add the relearned static rows to every run whose pieces are all in, and return the rest."""
    parts, missing = folder / "parts_self", []
    for tag, seed, _ in runs:
        name = f"study_{tag}_seed{seed}"
        stems = [piece_name(tag, seed, size) + beta_part(beta) for size in sizes for beta in betas]
        if not all((parts / f"{stem}.json").exists() for stem in stems):
            missing.append(name)
            continue
        new_rows = [
            row
            for stem in stems
            for row in json.loads((parts / f"{stem}.json").read_bytes())["rows"]
        ]
        report = add_self_calibrated(json.loads((folder / f"{name}.json").read_bytes()), new_rows)
        (folder / f"{name}.json").write_bytes(json.dumps(report, indent=1).encode())
        (folder / f"{name}.csv").write_bytes(rows_csv(report["rows"]))
        with open(folder / f"{name}.txt", "a", encoding="utf-8") as printout:
            printout.write(
                "\nstatic_day_self and static_hour_self added: tables relearned under "
                "themselves (--calibration self), split by beta; comparisons recomputed.\n"
            )
    return missing


@app.local_entrypoint()
def main(
    only: str = "",
    extra: str = "",
    out: str = "",
    label: str = "study",
    collect: bool = False,
    self_calibration: bool = False,
):
    """Launch every piece not yet collected, or collect what the volume holds, then merge.

    Args:
        only: Comma-separated scenario tags (all when empty).
        extra: Flags added to every run, for a short check (e.g. "--test-days 1").
        out: Folder to write to instead of results/study/.
        label: The volume folder pieces are kept in; use another for a check.
        collect: Download the finished pieces from the volume instead of launching.
        self_calibration: Run static_day and static_hour relearned under themselves, split
            by beta, and add them to the runs already merged as static_day_self and
            static_hour_self.
    """
    folder = Path(out) if out else STUDY
    if self_calibration and label == "study":
        label = "self"
    parts = folder / ("parts_self" if self_calibration else "parts")
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
        if self_calibration:
            relearn = ["--calibration", "self", "--policies", "recserve," + ",".join(STATIC)]
            jobs = [
                (
                    tag,
                    seed,
                    flags + relearn + ["--betas", f"{beta:g}"],
                    size,
                    label,
                    beta_part(beta),
                )
                for tag, seed, flags in runs
                for size in sizes
                for beta in study_betas()
                if not (parts / f"{piece_name(tag, seed, size)}{beta_part(beta)}.json").exists()
            ]
        else:
            jobs = [
                (tag, seed, flags, size, label, "")
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

    if self_calibration:
        missing = splice_finished(runs, sizes, study_betas(), folder)
    else:
        missing = merge_finished(runs, sizes, folder)
    print(f"{len(runs) - len(missing)} of {len(runs)} runs finished in {folder}")
    if missing:
        print("still missing pieces: " + ", ".join(missing))
