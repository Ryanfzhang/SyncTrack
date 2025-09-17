#! /bin/bash
#SBATCH --job-name=Bio
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH -p normal
#SBATCH --mail-user=mafzhang@ust.hk
#SBATCH --time=72:00:00
#SBATCH --account=earthhk
#SBATCH --output=log_sbatch/exp4.out
#SBATCH --cpus-per-gpu=4

###############################

python train_musicldm.py --config config/MSG-LD/multichannel_musicldm_slakh_3d_train.yaml

###############################