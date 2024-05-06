#!/bin/bash

# CONSTANT SLURM VALUES
#SBATCH --job-name=nanoGPT-training
#SBATCH --output=nanoGPT_%j.out
#SBATCH --error=nanoGPT_%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive

# SCRIPT USAGE
# `sbatch --partition=<YOUR_PARTITION_NAME> --nodes=<YOUR_NUMBER_OF_NODES> --nodelist=<HOSTNAME_ONE,HOSTNAME_TWO,...> --export=MASTER_NODE=<MASTER_HOSTNAME> nanoGPT_slurm.sh`

srun --ntasks-per-node=1 --exclusive bash -c '
HOSTNAME=$(hostname)

# Source the `conda.sh` file, this is just an example - your path may be different
source /opt/anaconda3/etc/profile.d/conda.sh

# Activate your nanoGPT conda environment by name - your name may be different
conda activate nanoGPT

# Change to the nanoGPT directory that contains the `train.py` file
cd /path/to/your/nanoGPT/repo_dir

# Function to define Pytorch commands, add your master node IP address and preferred port
run_torch_command() {
  NODE_RANK=$1
  MASTER_ADDR="YOUR_MASTER_NODE_IP_ADDRESS"
  MASTER_PORT="YOUR_MASTER_NODE_IP_PORT"
  NCCL_SETTINGS="NCCL_IB_PCI_RELAXED_ORDERING=1 NCCL_IB_GID_INDEX=3 HSA_FORCE_FINE_GRAIN_PCIE=1 NCCL_PROTO=Simple"
  COMMAND="$NCCL_SETTINGS torchrun --nproc_per_node=8 --nnodes=${SLURM_NNODES} --node_rank=$NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py config/train_gpt2.py --compile=False"

  echo "Running on $HOSTNAME with node rank $NODE_RANK"
  eval $COMMAND
}

if [ "$HOSTNAME" == "$MASTER_NODE" ]; then
    NODE_RANK=0
else
    NODE_RANK=$((SLURM_NODEID + 1))
fi

run_torch_command $NODE_RANK

# Deactivate the conda environment after the task completes
conda deactivate'
