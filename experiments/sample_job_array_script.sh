#!/bin/sh
#SBATCH --verbose
#SBATCH -p aquila
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=12GB
#SBATCH --mail-type=ALL # select which email types will be sent
#SBATCH --mail-user=netid@nyu.edu # put your netid here if you want emails

# #SBATCH --gres=gpu:1 # use this line to request for a gpu if your program uses gpu

#SBATCH --array=0-49 # here the number depends on number of jobs in the array
#SBATCH --output=sbl_%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID
#SBATCH --error=sbl_%A_%a.err

#SBATCH --constraint=cpu # use this if you want to only use cpu

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module load anaconda3 gcc/7.3 glfw/3.3 
module load mesa/19.0.5
module load llvm/7.0.1
source activate rl

echo ${SLURM_ARRAY_TASK_ID}
python sample_job_array_grid.py --setting ${SLURM_ARRAY_TASK_ID}
