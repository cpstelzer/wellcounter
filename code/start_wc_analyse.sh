#!/bin/bash 

cd $SCRATCH/wellcounter/popgrowth_20240403

dates=(20250701 20250702 20250703 20250704 20250705 20250706 20250707 20250708 20250709)

for date in "${dates[@]}"
do
   sbatch $SCRATCH/wellcounter/popgrowth_20250627/wc_analyse_date.slrm $date 
done

