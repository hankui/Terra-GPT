#!/bin/bash
#SBATCH --job-name=hls.gpt.app # Job name
#SBATCH --nodes=1 # Number of nodes
#SBATCH --mem 507904M # in unit MB 507904M=496G; is the max of the innovator 
#SBATCH --gres=gpu:2   # Number of GPUs per node (MAX is 2 for SDSU tested on Hank on Apr 19 2021)
#SBATCH --ntasks-per-node=1 # CPUs per node (MAX=40 for CPU nodes and 80 for GPU)
#SBATCH --output=out-%j-%N.log # Standard output (log file)
#SBATCH --partition=all-gpu # Partition/Queue
#SBATCH --time=2-00:00:00 # Maximum walltime 192=8-days
#SBATCH --mail-type=ALL #see https://slurm.schedmd.com/sbatch.html for all options.
#SBATCH --mail-user=junjie.li@sdstate.edu

# cd /mmfs1/home/jacks.local/hankui.zhang/mycode.research/LCMAP/HLS-GPT-tasks/HLS-GPT-tasks-02.27.2026
# sbatch my.hls.gpt.application.gpu.slurm
# squeue -l | grep hankui
# nvidia-smi

# bash my.hls.gpt.application.gpu.slurm

module load tensorflow/2.15
export HDF5_USE_FILE_LOCKING=FALSE

# module load rasterio/tensorflow # changed on Mar 5, 2024 
which python
export LD_LIBRARY_PATH=/mmfs1/cm/shared/apps_local/cudnn/8.9.7-cuda12/lib:$LD_LIBRARY_PATH
# conda init bash 
date_start=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;


version=4_1
version=4_3
tile_id='11SPS'
PREDICT_DATES='2023195,2023220'   # multi days
# PREDICT_DATES='2023195'
start_date=2023001
hls_data_dir='/mmfs1/scratch/jacks.local/junjie.li/HLS_reconstract/test_data/'
output_dir='/mmfs1/scratch/jacks.local/hankui.zhang/test/'


version=4_3
tile_id='11SLT' # The California fire for fuel moisture 
PREDICT_DATES='2024009,2024052,2024112,2024172,2024232,2024292,2024352,2024359,2024366,2025007'   # multi days
PREDICT_DATES='2024001,,2024352,2024359,2024366,2025007'   # multi days
start_date=2024001
hls_data_dir='/mmfs1/scratch/jacks.local/junjie.li/HLS_reconstract/test_data/'
output_dir='/mmfs1/scratch/jacks.local/hankui.zhang/test/'



version=4_3
tile_id='15TVH' # The Iowa soil moisture, tile: 15TVH, 15TVG
PREDICT_DATES=''   # multi days
start_date=2016001
hls_data_dir='/mmfs1/scratch/jacks.local/junjie.li/SoilMoisture2026/Iowa_HLS/'
output_dir='/mmfs1/scratch/jacks.local/junjie.li/SoilMoisture2026/Iowa_SM_output/'


version=4_4
tile_id='14TNP' # nasa-ibm crop mapping
PREDICT_DATES=''   #
start_date=2023001
hls_data_dir='/mmfs1/scratch/jacks.local/junjie.li/HLS_reconstract/test_data/'
output_dir='/mmfs1/scratch/jacks.local/junjie.li/HLS_reconstract/test_crop_mapping/'


date_start=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
python Pro_HLS_GPT_application_v${version}.py ${tile_id} ${start_date} ${hls_data_dir} ${output_dir} ${PREDICT_DATES} > Cropmapping.TILE${tile_id}.DATES${PREDICT_DATES:0:7}.txt

date_end=`date|awk -F"[ :]+" '{print $3*3600*24 + $4*60*60 + $5*60 + $6}'`;
time_diff=`echo "scale=2;($date_end-$date_start+0.01)*1.0/3600.0"|bc`;
date
echo "$time_diff hours used";

