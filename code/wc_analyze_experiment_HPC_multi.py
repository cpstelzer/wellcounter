# -*- coding: utf-8 -*-
"""Cluster-optimized driver for the Wellcounter experiment analysis.

This script mirrors the logic of ``wc_analyze_experiment.py`` while keeping the
paths and configuration flexible for execution on a high-performance computing
cluster.  It supports multiple acquisition dates stored in individual run
folders (``YYYYMMDD_batchX_plateY_wellZ``) inside an ``image_sequences``
directory and appends the combined particle/male summaries (via
``wim.count_complete``) to a CSV file for downstream analysis.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

import pandas as pd

import wellcounter_imaging_module as wim
import wellcounter_motion_module as wmm


def discover_dates(run_root: Path) -> Iterable[str]:
    """Return the sorted acquisition dates detected inside ``run_root``."""

    if not run_root.exists():
        raise FileNotFoundError(f"Data directory not found: {run_root}")

    date_prefixes = {
        folder.name.split("_")[0]
        for folder in run_root.iterdir()
        if folder.is_dir() and "_" in folder.name
    }
    return sorted(date_prefixes)


def load_processed_entries(output_csv: Path) -> Set[Tuple[str, int, int, int]]:
    """Load already processed (date, batch, plate, well) combinations."""

    if not output_csv.exists():
        return set()

    processed_df = pd.read_csv(output_csv)
    required_columns = {"date", "batch", "plate", "well"}
    if not required_columns.issubset(processed_df.columns):
        return set()

    return {
        (
            str(row["date"]),
            int(row["batch"]),
            int(row["plate"]),
            int(row["well"]),
        )
        for _, row in processed_df.iterrows()
    }


def analyze_experiment(
    experiment_root: Path,
    *,
    image_subdir: str = "image_sequences",
    treatments_filename: Optional[str] = None,
    output_filename: Optional[str] = None,
    include_motion: bool = False,
    resume: bool = False,
) -> None:
    """Run the multi-date experiment analysis on the cluster."""

    experiment_root = experiment_root.expanduser().resolve()

    if not experiment_root.exists():
        raise FileNotFoundError(f"Experiment directory not found: {experiment_root}")
    data_dir = experiment_root / image_subdir

    if treatments_filename is None:
        treatments_filename = f"{experiment_root.name}_treatments.csv"
    treat_path = experiment_root / treatments_filename

    if output_filename is None:
        output_filename = f"{experiment_root.name}_results.csv"
    outpath = experiment_root / output_filename

    print(f"Experiment root: {experiment_root}")
    print(f"Treatments file: {treat_path}")
    print(f"Output file:     {outpath}")
    print(f"Image directory: {data_dir}\n")

    if not treat_path.exists():
        raise FileNotFoundError(f"Treatments file not found: {treat_path}")

    if outpath.exists() and not resume:
        print(f"Existing output removed (fresh run): {outpath}")
        outpath.unlink()

    treat_df = pd.read_csv(treat_path)
    processed_entries = load_processed_entries(outpath)
    known_dates = discover_dates(data_dir)

    if not known_dates:
        print("No acquisition dates detected — nothing to process.")
        return

    print(f"Discovered acquisition dates: {', '.join(known_dates)}")

    for _, row in treat_df.iterrows():
        batch_no = int(row["batch"])
        plate_no = int(row["plate"])
        well_no = int(row["well"])

        print("\n" + "-" * 60)
        print(
            f"Processing sample: batch={batch_no}, plate={plate_no}, "
            f"well={well_no}"
        )
        print("-" * 60)

        found_folder = False

        for date_str in known_dates:
            identifier = (date_str, batch_no, plate_no, well_no)
            if identifier in processed_entries:
                print(
                    f"Skipping already processed folder for date {date_str}: "
                    f"batch{batch_no}_plate{plate_no}_well{well_no}"
                )
                continue

            folder_name = (
                f"{date_str}_batch{batch_no}_plate{plate_no}_well{well_no}"
            )
            run_folder_path = data_dir / folder_name

            if not run_folder_path.is_dir():
                continue

            found_folder = True

            print("\n" + "=" * 50)
            print(f"Analyzing: {folder_name}")
            print("=" * 50)

            try:
                count_df, _frame_stats_df, _joined_df = wim.count_complete(
                    str(run_folder_path)
                )

                current_row_df = row.to_frame().T
                current_row_df["date"] = date_str

                dfs_to_concat = [current_row_df.reset_index(drop=True), count_df]

                if include_motion:
                    motion_df = wmm.perform_motion_analysis(str(run_folder_path))
                    dfs_to_concat.append(motion_df.reset_index(drop=True))

                concatenated_df = pd.concat(dfs_to_concat, axis=1)

                header = not outpath.exists()
                concatenated_df.to_csv(outpath, mode="a", index=False, header=header)

                processed_entries.add(identifier)

            except Exception as exc:  # noqa: BLE001
                print(f"Error analyzing {folder_name}: {exc}")

        if not found_folder:
            print(
                "Warning: No data folders found for batch "
                f"{batch_no}, plate {plate_no}, well {well_no}."
            )

    print("\nAnalysis complete. Results saved to:", outpath)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Wellcounter experiments on the HPC cluster."
    )
    parser.add_argument(
        "experiment_name",
        help=(
            "Name of the experiment folder below the base directory. "
            "For example: 'popgrowth_20251022'."
        ),
    )
    parser.add_argument(
        "--base-dir",
        default=os.environ.get("SCRATCH", str(Path.cwd())),
        help="Base directory that contains the 'wellcounter' workspace.",
    )
    parser.add_argument(
        "--image-subdir",
        default="image_sequences",
        help="Subdirectory that stores per-run folders (default: image_sequences).",
    )
    parser.add_argument(
        "--treatments-filename",
        default=None,
        help="Custom treatments CSV filename (defaults to '<experiment>_treatments.csv').",
    )
    parser.add_argument(
        "--output-filename",
        default=None,
        help="Custom output CSV filename (defaults to '<experiment>_results.csv').",
    )
    parser.add_argument(
        "--include-motion",
        action="store_true",
        help="Enable motion analysis via wellcounter_motion_module.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to an existing results file instead of recreating it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir).expanduser().resolve()
    experiment_root = base_dir / "wellcounter" / args.experiment_name

    analyze_experiment(
        experiment_root,
        image_subdir=args.image_subdir,
        treatments_filename=args.treatments_filename,
        output_filename=args.output_filename,
        include_motion=args.include_motion,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
