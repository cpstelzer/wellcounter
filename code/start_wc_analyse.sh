#!/bin/bash

cd "$SCRATCH/wellcounter/popgrowth_20251001"

dates=(
    20251002
    20251003
    20251004
    20251005
    20251006
    20251007
    20251008
    20251009
    20251010
)

for date in "${dates[@]}"
do
   sbatch "$SCRATCH/wellcounter/popgrowth_20250627/wc_analyse_date.slrm" "$date"
done
