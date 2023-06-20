#!/bin/bash -l
#SBATCH -J diffusion_grasp_eval_63cat
#SBATCH --partition=gpu 
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 12                # Cores assigned to each tasks
#SBATCH --time 2-00:00:00
#SBATCH -G 1
#SBATCH -C volta
#SBATCH -o %x-%j.out # ./<jobname>-<jobid>.out

print_error_and_exit() { echo "***ERROR*** $*"; exit 1; }
module purge || print_error_and_exit "No 'module' command"

# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export MODULEPATH=/opt/apps/resif/iris/2020b/gpu/modules/all:/opt/apps/resif/iris/2020b/skylake/modules/all/

module load system/CUDA/11.1.1-GCC-10.2.0
module load numlib/cuDNN/8.0.5.39-CUDA-11.1.1
module load numlib/SuiteSparse/5.8.1-foss-2020b-METIS-5.1.0

conda activate se3dif_env

ulimit -n 50000

cd ~/grasp/grasp_diffusion

LD_LIBRARY_PATH=~/miniconda3/envs/se3dif_env/lib python scripts/evaluate/evaluate_multiple.py --n_grasps 100 --model grasp_dif_multi --device cuda:0 --n_envs 100