"""Utilities for Stage 02 particle characterization outputs.

This module augments the Stage 02 workflow with a final step that selects the
particle whose ``main_area_filled`` value lies at the median of the
``final_particles`` table. The selected particle is exported as:

1. A cropped image focused on its bounding box.
2. A CSV file containing all attributes of the median particle.

Both artifacts are written to ``stage2_characterization/final_median``.

The helper functions intentionally avoid assumptions about how Stage 02 stores
its intermediate files. They look for sensible defaults (for example,
``stage2_characterization/final_particles/final_particles.csv``) but also allow
paths to be provided explicitly.  Bounding boxes can be stored either as
``bounding_x``, ``bounding_y``, ``bounding_w`` and ``bounding_h`` columns (as
produced by :func:`wellcounter_imaging_module.calculate_measurements`) or as
``min/max`` coordinate pairs (``xmin``, ``ymin``, ``xmax``, ``ymax`` and common
variations).  Likewise, the source image can be specified in the particle row
via ``image_path``/``frame_path``/``source_image`` columns or provided as a
function argument.

Example
-------
Run the module as a script once Stage 02 produced ``final_particles``::

    python code/stage2_characterization.py \
        --stage2-root /data/stage2_characterization \
        --final-particles final_particles/final_particles.csv \
        --reference-image reference_images/frame1_particles.jpg

Relative paths for ``--final-particles`` and ``--reference-image`` are resolved
from ``--stage2-root`` so that all inputs remain inside the
``stage2_characterization`` folder.  The command writes the cropped median
particle image and metadata CSV into ``stage2_characterization/final_median``.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2
import pandas as pd

IMAGE_EXTENSIONS: Sequence[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
FINAL_MEDIAN_SUBDIR = "final_median"
DEFAULT_STAGE2_ROOT = Path("stage2_characterization")


def stage2_save_median_particle(
    stage2_root: str | Path = DEFAULT_STAGE2_ROOT,
    final_particles_path: Optional[str | Path] = None,
    reference_image_path: Optional[str | Path] = None,
    area_column: str = "main_area_filled",
    padding: int = 10,
) -> Tuple[Path, Path]:
    """Export the median particle artifacts for Stage 02.

    Parameters
    ----------
    stage2_root:
        Folder that contains Stage 02 outputs (``stage2_characterization`` by
        default).
    final_particles_path:
        Optional path (absolute or relative to ``stage2_root``) that points to
        the CSV file with the ``final_particles`` table. When omitted, the
        function searches for a file named ``final_particles*.csv`` under
        ``stage2_root``.
    reference_image_path:
        Path to the image that contains the bounding boxes referenced by
        ``final_particles``.  Relative paths are interpreted with respect to
        ``stage2_root`` to keep all inputs inside the Stage 02 directory. When
        omitted, the function attempts to infer the image path from the
        particle table or by scanning the Stage 02 folder.
    area_column:
        Name of the column that stores the ``main_area_filled`` measurement.
    padding:
        Extra pixels added to every side of the bounding box when cropping.

    Returns
    -------
    tuple(Path, Path)
        Paths to the cropped image and CSV metadata that were written.
    """

    stage2_root = Path(stage2_root)
    final_particles_file = _locate_final_particles_file(stage2_root, final_particles_path)
    particles_df = _load_particles(final_particles_file)

    median_row, median_value = _select_median_row(particles_df, area_column)
    bbox = _resolve_bounding_box(median_row)
    reference_image = _resolve_reference_image(
        median_row, stage2_root, reference_image_path, final_particles_file.parent
    )
    cropped_image = _crop_particle_image(reference_image, bbox, padding)

    output_dir = stage2_root / FINAL_MEDIAN_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = _determine_output_basename(median_row, final_particles_file)
    image_output_path = output_dir / f"{base_name}_median_particle.png"
    csv_output_path = output_dir / f"{base_name}_median_particle.csv"

    if not cv2.imwrite(str(image_output_path), cropped_image):
        raise RuntimeError(f"Failed to save cropped image to {image_output_path}")

    metadata_df = median_row.to_frame().T.copy()
    metadata_df["median_main_area_filled"] = median_value
    metadata_df["median_particle_image"] = image_output_path.name
    metadata_df.to_csv(csv_output_path, index=False)

    print(
        "Median particle artifacts saved to",
        f"{image_output_path} and {csv_output_path}",
    )
    return image_output_path, csv_output_path


def _locate_final_particles_file(stage2_root: Path, explicit_path: Optional[str | Path]) -> Path:
    if explicit_path:
        candidate = Path(explicit_path)
        if not candidate.is_absolute():
            candidate = stage2_root / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(
                f"Could not find final_particles file at {candidate}."
            )
        return candidate

    if not stage2_root.exists():
        raise FileNotFoundError(
            f"Stage 02 root '{stage2_root}' does not exist."
        )

    # Try common default locations first.
    default_locations = [
        stage2_root / "final_particles" / "final_particles.csv",
        stage2_root / "final_particles.csv",
    ]
    for location in default_locations:
        if location.exists():
            return location.resolve()

    matches = sorted(stage2_root.rglob("final_particles*.csv"))
    if matches:
        return matches[0].resolve()

    raise FileNotFoundError(
        "Unable to locate a final_particles CSV inside Stage 02 outputs."
    )


def _load_particles(final_particles_file: Path) -> pd.DataFrame:
    df = pd.read_csv(final_particles_file)
    if df.empty:
        raise ValueError(
            f"The final_particles file '{final_particles_file}' does not contain any rows."
        )
    return df


def _select_median_row(df: pd.DataFrame, area_column: str) -> Tuple[pd.Series, float]:
    if area_column not in df.columns:
        raise KeyError(
            f"Column '{area_column}' is missing from the final_particles table."
        )

    valid_df = df.dropna(subset=[area_column])
    if valid_df.empty:
        raise ValueError(
            f"Column '{area_column}' does not contain any valid values."
        )

    median_value = float(valid_df[area_column].median())
    closest_idx = (valid_df[area_column] - median_value).abs().idxmin()
    return df.loc[closest_idx], median_value


def _resolve_bounding_box(row: pd.Series) -> Tuple[int, int, int, int]:
    bounding_columns = {"bounding_x", "bounding_y", "bounding_w", "bounding_h"}
    if bounding_columns.issubset(row.index):
        x_min = int(row["bounding_x"])
        y_min = int(row["bounding_y"])
        x_max = x_min + int(row["bounding_w"])
        y_max = y_min + int(row["bounding_h"])
        return x_min, y_min, x_max, y_max

    candidate_sets = [
        ("xmin", "ymin", "xmax", "ymax"),
        ("x_min", "y_min", "x_max", "y_max"),
        ("min_col", "min_row", "max_col", "max_row"),
        ("bbox_min_col", "bbox_min_row", "bbox_max_col", "bbox_max_row"),
        ("left", "top", "right", "bottom"),
    ]
    for x_min_key, y_min_key, x_max_key, y_max_key in candidate_sets:
        if {x_min_key, y_min_key, x_max_key, y_max_key}.issubset(row.index):
            return (
                int(row[x_min_key]),
                int(row[y_min_key]),
                int(row[x_max_key]),
                int(row[y_max_key]),
            )

    raise KeyError(
        "Could not determine bounding box columns in final_particles."
    )


def _resolve_reference_image(
    row: pd.Series,
    stage2_root: Path,
    explicit_path: Optional[str | Path],
    fallback_dir: Path,
) -> Path:
    if explicit_path:
        candidate = Path(explicit_path)
        if not candidate.is_absolute():
            candidate = stage2_root / candidate
        candidate = candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Reference image '{candidate}' not found.")
        return candidate

    for column in ("image_path", "frame_path", "source_image", "parent_image"):
        if column in row and isinstance(row[column], str) and row[column].strip():
            candidate = Path(row[column])
            if not candidate.is_absolute():
                candidate = (stage2_root / candidate).resolve()
            if candidate.exists():
                return candidate

    for search_root in (fallback_dir, stage2_root):
        for extension in IMAGE_EXTENSIONS:
            matches = sorted(search_root.glob(f"*{extension}"))
            if matches:
                return matches[0].resolve()

    raise FileNotFoundError(
        "Unable to locate a reference image for cropping the median particle."
    )


def _crop_particle_image(
    image_path: Path, bbox: Tuple[int, int, int, int], padding: int
):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image '{image_path}'.")

    height, width = image.shape[:2]
    x_min, y_min, x_max, y_max = bbox

    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(width, x_max + padding)
    y_max = min(height, y_max + padding)

    if x_min >= x_max or y_min >= y_max:
        raise ValueError("Bounding box has zero or negative area after padding.")

    return image[y_min:y_max, x_min:x_max]


def _determine_output_basename(row: pd.Series, final_particles_file: Path) -> str:
    for candidate_key in ("particle_id", "particleID", "id"):
        if candidate_key in row and str(row[candidate_key]).strip():
            return _sanitize_identifier(str(row[candidate_key]))

    if row.name is not None:
        return _sanitize_identifier(f"particle_{row.name}")

    return _sanitize_identifier(final_particles_file.stem)


def _sanitize_identifier(value: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_-]+", "_", value).strip("_")
    return cleaned or "median_particle"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize Stage 02 characterization outputs by exporting the median "
            "particle's image crop and metadata."
        )
    )
    parser.add_argument(
        "--stage2-root",
        default=str(DEFAULT_STAGE2_ROOT),
        help="Directory that stores Stage 02 outputs (default: stage2_characterization).",
    )
    parser.add_argument(
        "--final-particles",
        dest="final_particles",
        default=None,
        help=(
            "Path to final_particles CSV relative to the Stage 02 root. If omitted, "
            "a file named final_particles*.csv is searched."
        ),
    )
    parser.add_argument(
        "--reference-image",
        dest="reference_image",
        default=None,
        help=(
            "Explicit path to the reference image that will be cropped. Relative "
            "paths are interpreted from the Stage 02 root."
        ),
    )
    parser.add_argument(
        "--area-column",
        dest="area_column",
        default="main_area_filled",
        help="Name of the column with the area measurement (default: main_area_filled).",
    )
    parser.add_argument(
        "--padding",
        dest="padding",
        type=int,
        default=10,
        help="Padding (in pixels) to add around the bounding box (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stage2_save_median_particle(
        stage2_root=args.stage2_root,
        final_particles_path=args.final_particles,
        reference_image_path=args.reference_image,
        area_column=args.area_column,
        padding=args.padding,
    )


if __name__ == "__main__":
    main()
