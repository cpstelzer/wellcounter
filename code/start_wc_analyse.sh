#!/bin/bash

# This script must be checked out with Unix (LF) line endings.
# See .gitattributes if you encounter $'\r' errors when running it.
set -euo pipefail 2>/dev/null || set -eu

if [ $# -lt 1 ]; then
    echo "Usage: $(basename "$0") <experiment_name> [additional python args]" >&2
    exit 1
fi

if [ -z "${SCRATCH:-}" ]; then
    echo "Environment variable SCRATCH is not defined." >&2
    exit 1
fi

experiment_name="$1"
shift

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
slurm_script="${script_dir}/wc_analyse_date.slrm"

if [ ! -f "$slurm_script" ]; then
    echo "Unable to locate SLURM script: $slurm_script" >&2
    exit 1
fi

scratch_root="$SCRATCH/wellcounter"
if [ ! -d "$scratch_root" ]; then
    echo "Expected project directory not found: $scratch_root" >&2
    exit 1
fi

echo "Submitting Wellcounter analysis for experiment '${experiment_name}'."
sbatch "$slurm_script" "$experiment_name" "$@"

