# DeepSpeed Multi-Node Setup Guide

A comprehensive guide to setting up and running DeepSpeed training across multiple nodes, covering cluster configuration, network optimization, and debugging.

---

## Table of Contents

1. [Introduction to Multi-Node Training](#introduction-to-multi-node-training)
2. [Prerequisites](#prerequisites)
3. [Cluster Setup](#cluster-setup)
4. [SSH Configuration](#ssh-configuration)
5. [Hostfile Configuration](#hostfile-configuration)
6. [Network Optimization](#network-optimization)
7. [Launching Multi-Node Jobs](#launching-multi-node-jobs)
8. [SLURM Integration](#slurm-integration)
9. [Debugging Multi-Node Issues](#debugging-multi-node-issues)
10. [Best Practices](#best-practices)

---

## Introduction to Multi-Node Training

### Why Multi-Node?

**Single Node Limitations**:
- Limited to 8 GPUs typically
- Memory capacity capped at ~640GB (8× A100 80GB)
- Can't train models > 175B parameters efficiently

**Multi-Node Benefits**:
- Scale to hundreds of GPUs
- Train massive models (1T+ parameters)
- Faster training through parallelism
- Cost-effective with spot instances

### Communication Challenges

Multi-node training faces unique challenges:
- **Bandwidth**: Inter-node ~10-25 GB/s vs intra-node ~600 GB/s (NVLink)
- **Latency**: Higher latency between nodes
- **Reliability**: More failure points
- **Synchronization**: Keeping nodes in sync

**DeepSpeed Solutions**:
- ZeRO optimizations reduce communication
- Gradient compression (1-bit Adam)
- Pipeline parallelism minimizes inter-node traffic
- Robust fault tolerance

---

## Prerequisites

### Hardware Requirements

**Minimum**:
- 2+ compute nodes
- 1+ GPU per node (ideally 8 GPUs/node)
- High-speed interconnect (10+ Gbps)
- Shared file system (NFS, Lustre, GPFS)

**Recommended**:
- 4-16 nodes
- 8 GPUs per node (NVIDIA A100, H100)
- InfiniBand (100+ Gbps)
- Fast shared storage (Lustre, BeeGFS)

**Network Topology**:
```
┌────────────────────────────────────────────────┐
│              Head Node (Login)                 │
│         - Job submission                       │
│         - NFS server (optional)                │
└────────────────┬───────────────────────────────┘
                 │
        ┌────────┴────────┐
        │   Ethernet      │
        │   or IB Switch  │
        └────────┬────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───▼────┐  ┌───▼────┐  ┌───▼────┐
│ Node 1 │  │ Node 2 │  │ Node 3 │
│ 8×GPU  │  │ 8×GPU  │  │ 8×GPU  │
└────────┘  └────────┘  └────────┘
```

### Software Requirements

**On All Nodes**:
- Same OS (Ubuntu 20.04/22.04 or RHEL 8)
- Same Python version (3.8+)
- Same CUDA version (11.8 or 12.1+)
- Same PyTorch version
- Same DeepSpeed version
- Passwordless SSH between nodes

**Verification**:
```bash
# Check versions on all nodes
pdsh -w node[1-4] "python --version"
pdsh -w node[1-4] "nvcc --version"
pdsh -w node[1-4] "python -c 'import torch; print(torch.__version__)'"
pdsh -w node[1-4] "python -c 'import deepspeed; print(deepspeed.__version__)'"
```

---

## Cluster Setup

### Step 1: Install Dependencies on All Nodes

**Using Ansible (Recommended)**:

Create `install_deps.yml`:
```yaml
---
- hosts: compute_nodes
  become: yes
  tasks:
    - name: Install system packages
      apt:
        name:
          - build-essential
          - python3-dev
          - python3-pip
          - libaio-dev
          - pdsh
        state: present
        update_cache: yes

    - name: Install PyTorch
      pip:
        name:
          - torch==2.1.0
          - torchvision
        executable: pip3

    - name: Install DeepSpeed
      pip:
        name: deepspeed
        executable: pip3

    - name: Build DeepSpeed ops
      shell: |
        DS_BUILD_OPS=1 pip install deepspeed --force-reinstall --no-cache-dir
      environment:
        CUDA_HOME: /usr/local/cuda
```

Run:
```bash
ansible-playbook -i hosts.ini install_deps.yml
```

**Manual Installation** (if no Ansible):
```bash
# On each node
sudo apt-get update
sudo apt-get install -y build-essential python3-dev libaio-dev pdsh

pip install torch==2.1.0
DS_BUILD_OPS=1 pip install deepspeed
```

---

### Step 2: Configure Shared File System

**Option A: NFS (Simple)**

On head node:
```bash
# Install NFS server
sudo apt-get install nfs-kernel-server

# Create shared directory
sudo mkdir -p /shared
sudo chown $USER:$USER /shared

# Export directory
echo "/shared *(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports
sudo exportfs -a
sudo systemctl restart nfs-kernel-server
```

On compute nodes:
```bash
# Install NFS client
sudo apt-get install nfs-common

# Mount shared directory
sudo mkdir -p /shared
sudo mount head-node:/shared /shared

# Make permanent
echo "head-node:/shared /shared nfs defaults 0 0" | sudo tee -a /etc/fstab
```

**Option B: Lustre (High Performance)**

Requires dedicated setup - consult cluster administrator.

---

### Step 3: Synchronize Environments

**Option 1: Shared Conda Environment**
```bash
# On head node (in /shared)
cd /shared
conda create -p ./deepspeed_env python=3.10
conda activate ./deepspeed_env
pip install torch deepspeed transformers

# On compute nodes
conda activate /shared/deepspeed_env
```

**Option 2: Container (Recommended for Production)**
```bash
# Build Singularity container
singularity build deepspeed.sif docker://deepspeed/deepspeed:latest

# Use on all nodes
singularity exec --nv deepspeed.sif python train.py
```

---

## SSH Configuration

### Passwordless SSH Setup

**On head node**:
```bash
# Generate SSH key (if not exists)
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# Copy to all compute nodes
for node in node1 node2 node3 node4; do
    ssh-copy-id $node
done

# Test
for node in node1 node2 node3 node4; do
    ssh $node hostname
done
```

**Expected output**:
```
node1
node2
node3
node4
```

### SSH Config for Convenience

Create `~/.ssh/config`:
```
Host node*
    StrictHostKeyChecking no
    UserKnownHostsFile=/dev/null
    ConnectTimeout=10
    ServerAliveInterval=60
    ServerAliveCountMax=3
```

### Verify Connectivity

```bash
# Test SSH to all nodes
pdsh -w node[1-4] hostname

# Test SSH with specific user
pdsh -w node[1-4] -l username hostname
```

---

## Hostfile Configuration

### Basic Hostfile

Create `hostfile`:
```
node1 slots=8
node2 slots=8
node3 slots=8
node4 slots=8
```

**Format**:
- `node1`: Hostname or IP
- `slots=8`: Number of GPUs on this node

### Advanced Hostfile

With specific network interfaces:
```
node1 slots=8 ib0
node2 slots=8 ib0
node3 slots=8 ib0
node4 slots=8 ib0
```

With different GPU counts:
```
node1 slots=8
node2 slots=4
node3 slots=8
```

### Environment Variables in Hostfile

Some clusters need environment setup:
```bash
# hostfile with env vars
node1 slots=8 NCCL_SOCKET_IFNAME=ib0
node2 slots=8 NCCL_SOCKET_IFNAME=ib0
```

---

## Network Optimization

### InfiniBand Configuration

**Check InfiniBand Status**:
```bash
# List IB devices
ibstat

# Check IB link
ibstatus

# Test IB bandwidth
ib_write_bw
```

**NCCL Settings for InfiniBand**:
```bash
export NCCL_IB_DISABLE=0              # Enable IB
export NCCL_IB_HCA=mlx5_0             # IB device
export NCCL_SOCKET_IFNAME=ib0         # IB interface
export NCCL_NET_GDR_LEVEL=3           # GPU Direct RDMA
export NCCL_IB_GID_INDEX=3            # RoCE v2
export NCCL_IB_TIMEOUT=22             # Timeout
```

**Verify GPU Direct**:
```bash
# Check if GPU Direct is enabled
nvidia-smi topo -m

# Should show "SYS" or "PHB" for IB connection
```

---

### Ethernet Configuration

**NCCL Settings for Ethernet**:
```bash
export NCCL_SOCKET_IFNAME=eth0        # Ethernet interface
export NCCL_IB_DISABLE=1              # Disable IB
export NCCL_NET_GDR_LEVEL=0           # No GPU Direct
export NCCL_DEBUG=INFO                # Debug output
```

**Check Network Interface**:
```bash
# List interfaces
ip addr show

# Test bandwidth between nodes (on node1)
iperf3 -s

# On node2
iperf3 -c node1 -P 8
```

---

### NCCL Tuning

**Environment Variables**:
```bash
# Performance
export NCCL_BUFFSIZE=2097152          # Buffer size (2MB)
export NCCL_P2P_LEVEL=NVL             # Use NVLink
export NCCL_SHM_DISABLE=0             # Enable shared memory

# Debugging
export NCCL_DEBUG=INFO                # Debug level
export NCCL_DEBUG_SUBSYS=ALL          # Debug all subsystems

# Topology
export NCCL_TOPO_FILE=/path/to/topo.xml  # Custom topology
```

**Create NCCL Topology File** (Advanced):
```xml
<system version="1">
  <cpu numaid="0" affinity="0000ffff" arch="x86_64" vendor="GenuineIntel" familyid="6" modelid="85">
    <pci busid="0000:00:00.0">
      <gpu dev="0" sm="80" rank="0" gdr="1"/>
      <gpu dev="1" sm="80" rank="1" gdr="1"/>
    </pci>
  </cpu>
</system>
```

---

## Launching Multi-Node Jobs

### Method 1: DeepSpeed Launcher (Recommended)

**Basic Launch**:
```bash
deepspeed --hostfile=hostfile \
          --num_nodes=4 \
          --num_gpus=8 \
          train.py \
          --deepspeed_config=ds_config.json
```

**With Master Node Specification**:
```bash
deepspeed --hostfile=hostfile \
          --master_addr=node1 \
          --master_port=29500 \
          train.py \
          --deepspeed_config=ds_config.json
```

**With Environment Variables**:
```bash
deepspeed --hostfile=hostfile \
          --num_nodes=4 \
          --num_gpus=8 \
          --launcher=pdsh \
          --launcher_args="-S" \
          train.py \
          --deepspeed_config=ds_config.json
```

---

### Method 2: Manual Launch with pdsh

**Launch Script** (`launch_multi_node.sh`):
```bash
#!/bin/bash

export MASTER_ADDR=node1
export MASTER_PORT=29500
export NCCL_SOCKET_IFNAME=ib0

# Read hostfile and launch on each node
NODE_RANK=0
while read -r line; do
    NODE=$(echo $line | awk '{print $1}')
    SLOTS=$(echo $line | awk '{print $2}' | cut -d= -f2)

    # Launch on this node
    pdsh -w $NODE \
        "cd /shared/project && \
         RANK=$NODE_RANK \
         WORLD_SIZE=32 \
         MASTER_ADDR=$MASTER_ADDR \
         MASTER_PORT=$MASTER_PORT \
         python -m torch.distributed.run \
         --nproc_per_node=$SLOTS \
         --nnodes=4 \
         --node_rank=$NODE_RANK \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         train.py --deepspeed" &

    NODE_RANK=$((NODE_RANK + 1))
done < hostfile

wait
```

---

### Method 3: MPI Launch

**Using mpirun** (if MPI installed):
```bash
mpirun -np 32 \
       -hostfile hostfile \
       -x NCCL_SOCKET_IFNAME=ib0 \
       -x MASTER_ADDR=node1 \
       -x MASTER_PORT=29500 \
       python train.py --deepspeed
```

---

## SLURM Integration

### SLURM Job Script

Create `train.slurm`:
```bash
#!/bin/bash
#SBATCH --job-name=deepspeed-train
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Load modules
module load cuda/12.1
module load nccl/2.18

# Activate environment
source /shared/deepspeed_env/bin/activate

# Set environment variables
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO

# Generate hostfile from SLURM
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile
cat hostfile | awk '{print $1 " slots=8"}' > deepspeed_hostfile

# Get master node
MASTER_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29500

# Launch DeepSpeed
deepspeed --hostfile=deepspeed_hostfile \
          --num_nodes=$SLURM_NNODES \
          --num_gpus=8 \
          train.py \
          --deepspeed_config=ds_config.json
```

**Submit Job**:
```bash
sbatch train.slurm
```

**Monitor Job**:
```bash
# Check status
squeue -u $USER

# View output
tail -f logs/train_12345.out

# Cancel job
scancel 12345
```

---

### SLURM with Array Jobs

For hyperparameter search:
```bash
#!/bin/bash
#SBATCH --array=0-9
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8

# Different LR for each job
LRS=(1e-5 5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 5e-1)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}

deepspeed train.py --learning_rate=$LR
```

---

## Debugging Multi-Node Issues

### Issue 1: Nodes Can't Communicate

**Symptoms**:
```
[Rank 0] Waiting for other ranks...
[Rank 8] Connection timeout
```

**Debug Steps**:

1. **Test SSH**:
```bash
# From head node
for node in node1 node2 node3; do
    echo "Testing $node..."
    ssh $node "echo SSH works on \$(hostname)"
done
```

2. **Test Network**:
```bash
# Ping test
pdsh -w node[1-4] "ping -c 1 node1"

# Port test
nc -zv node1 29500
```

3. **Check Firewall**:
```bash
# Disable firewall (temporarily for testing)
sudo ufw disable

# Or open ports
sudo ufw allow 29500:29600/tcp
```

---

### Issue 2: NCCL Initialization Hangs

**Symptoms**:
```
[Rank 0] Initializing NCCL...
(hangs indefinitely)
```

**Debug**:
```bash
# Enable NCCL debug
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV

# Run training
deepspeed train.py

# Check output for errors
```

**Common fixes**:
```bash
# Try different network interface
export NCCL_SOCKET_IFNAME=eth0  # or ib0

# Increase timeout
export NCCL_TIMEOUT=1800

# Disable IB if problematic
export NCCL_IB_DISABLE=1
```

---

### Issue 3: Inconsistent Results Across Nodes

**Symptoms**:
```
Node 0: Loss = 2.453
Node 1: Loss = NaN
```

**Debug**:

1. **Check Data Loading**:
```python
# Ensure same random seed
torch.manual_seed(42 + rank)

# Ensure same data order
sampler = torch.utils.data.distributed.DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
    seed=42
)
```

2. **Verify Model Sync**:
```python
# After initialization
if rank == 0:
    # Print model hash
    model_hash = hash(tuple(p.data.sum().item() for p in model.parameters()))
    print(f"Model hash: {model_hash}")
```

3. **Check for Race Conditions**:
```python
# Add barriers
torch.distributed.barrier()  # Wait for all ranks

# Synchronize file I/O
if rank == 0:
    # Write config
    with open('config.json', 'w') as f:
        json.dump(config, f)

torch.distributed.barrier()  # Wait for rank 0 to write

# All ranks read
with open('config.json', 'r') as f:
    config = json.load(f)
```

---

### Issue 4: One Node Slower Than Others

**Symptoms**:
```
Node 0: 500ms/step
Node 1: 500ms/step
Node 2: 1500ms/step  ← Slow!
Node 3: 500ms/step
```

**Debug**:

1. **Check GPU Health**:
```bash
# On slow node
nvidia-smi

# Look for:
# - Throttling
# - Power limit
# - ECC errors
```

2. **Check CPU/Memory**:
```bash
# CPU usage
htop

# I/O wait
iostat -x 1

# Network
iftop -i ib0
```

3. **Check Data Loading**:
```python
# Profile data loading
import time

start = time.time()
for batch in dataloader:
    pass
elapsed = time.time() - start
print(f"Data loading: {elapsed:.2f}s")
```

---

## Best Practices

### 1. Use Hierarchical Communication

For >8 nodes, organize communication hierarchically:
```json
{
  "zero_optimization": {
    "stage": 3,
    "reduce_bucket_size": 5e8,
    "overlap_comm": true
  },
  "communication_data_type": "fp16",
  "pipeline": {
    "activation_checkpoint_interval": 1
  }
}
```

### 2. Monitor Training

**Use TensorBoard**:
```python
from torch.utils.tensorboard import SummaryWriter

if rank == 0:
    writer = SummaryWriter()

# Log only on rank 0
if rank == 0:
    writer.add_scalar('Loss/train', loss, step)
```

**Use Weights & Biases**:
```python
import wandb

if rank == 0:
    wandb.init(project="multi-node-training")

if rank == 0:
    wandb.log({"loss": loss, "step": step})
```

### 3. Implement Checkpointing

**Save checkpoints regularly**:
```python
# Save every N steps
if step % 1000 == 0:
    model_engine.save_checkpoint(
        save_dir='checkpoints',
        tag=f'step_{step}'
    )
```

**Implement restart logic**:
```python
# Find latest checkpoint
import glob
checkpoints = glob.glob('checkpoints/step_*')
if checkpoints:
    latest = max(checkpoints, key=lambda x: int(x.split('_')[-1]))
    model_engine.load_checkpoint(latest)
```

### 4. Handle Failures Gracefully

**Catch and log errors**:
```python
try:
    for step, batch in enumerate(dataloader):
        loss = model_engine(batch)
        model_engine.backward(loss)
        model_engine.step()
except Exception as e:
    print(f"Rank {rank} error: {e}")
    # Save emergency checkpoint
    model_engine.save_checkpoint('emergency', tag=f'rank_{rank}_error')
    raise
```

### 5. Optimize Batch Size

**Scale batch size with nodes**:
```python
# Base batch size for single node
base_batch_size = 32

# Scale with world size
world_size = torch.distributed.get_world_size()
batch_size = base_batch_size * (world_size // 8)  # 8 GPUs per node
```

---

## Performance Checklist

- [ ] **Network**: InfiniBand enabled and working
- [ ] **NCCL**: Correct environment variables set
- [ ] **Data Loading**: num_workers > 0, pin_memory=True
- [ ] **Batch Size**: Scaled appropriately with nodes
- [ ] **Communication**: overlap_comm=true in config
- [ ] **Checkpointing**: Regular checkpoints enabled
- [ ] **Monitoring**: Logging to TensorBoard/WandB
- [ ] **Shared Storage**: Fast shared file system
- [ ] **Environment**: Synchronized across nodes

---

## Example: Complete Multi-Node Training Script

```python
#!/usr/bin/env python
"""
Multi-node DeepSpeed training script.
Usage:
    deepspeed --hostfile=hostfile train_multinode.py
"""

import os
import torch
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer

def setup_environment():
    """Setup distributed training environment."""
    # Get rank info
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    print(f"[Rank {rank}/{world_size}] Local rank: {local_rank}")

    # Set device
    torch.cuda.set_device(local_rank)

    return local_rank, rank, world_size

def main():
    # Setup
    local_rank, rank, world_size = setup_environment()

    # Load model
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # DeepSpeed config
    ds_config = {
        "train_batch_size": 128 * (world_size // 8),  # Scale with nodes
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 4,
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": 1e-4}
        }
    }

    # Initialize DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )

    # Training loop
    model_engine.train()
    for step in range(1000):
        # Dummy batch (replace with real data)
        input_ids = torch.randint(0, 50000, (4, 512)).to(model_engine.device)
        labels = input_ids.clone()

        # Forward
        outputs = model_engine(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        # Backward and step
        model_engine.backward(loss)
        model_engine.step()

        # Log on rank 0
        if rank == 0 and step % 100 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

        # Checkpoint
        if step % 500 == 0:
            model_engine.save_checkpoint('checkpoints', tag=f'step_{step}')

    if rank == 0:
        print("Training complete!")

if __name__ == '__main__':
    main()
```

---

## Additional Resources

- **[DeepSpeed Multi-Node Tutorial](https://www.deepspeed.ai/getting-started/#multi-node-training)** - Official docs
- **[NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/)** - NCCL tuning guide
- **[SLURM Documentation](https://slurm.schedmd.com/)** - SLURM job scheduling
- **[InfiniBand Tuning](https://community.mellanox.com/)** - IB optimization

**Happy multi-node training!** 🚀
