#!/bin/sh
#SBATCH --verbose
#SBATCH -p aquila
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=16GB
#SBATCH --mail-type=ALL # select which email types will be sent
#SBATCH --mail-user=yw1370@nyu.edu


#SBATCH --array=0-449
#SBATCH --output=sbl_%A_%a.out # %A is SLURM_ARRAY_JOB_ID, %a is SLURM_ARRAY_TASK_ID
#SBATCH --error=sbl_%A_%a.err

##SBATCH --constraint=2630v3
#SBATCH --constraint=cpu


echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

module load anaconda3 gcc/7.3 glfw/3.3 
module load mesa/19.0.5
module load llvm/7.0.1
source activate rl

echo ${SLURM_ARRAY_TASK_ID}
python SOP_job_array.py --setting ${SLURM_ARRAY_TASK_ID}
