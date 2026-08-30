# DeepSpeed Cost Optimization Guide

A comprehensive guide to minimizing training costs with DeepSpeed, covering cloud provider pricing, spot instances, configuration tradeoffs, and ROI optimization.

---

## Table of Contents

1. [Understanding Training Costs](#understanding-training-costs)
2. [Cloud Provider Pricing](#cloud-provider-pricing)
3. [Spot Instance Strategies](#spot-instance-strategies)
4. [Configuration Cost Tradeoffs](#configuration-cost-tradeoffs)
5. [Memory vs Speed Optimization](#memory-vs-speed-optimization)
6. [Multi-Node Cost Considerations](#multi-node-cost-considerations)
7. [Cost Calculation Examples](#cost-calculation-examples)
8. [Cost Reduction Strategies](#cost-reduction-strategies)
9. [ROI Optimization](#roi-optimization)

---

## Understanding Training Costs

### Cost Components

**Direct Costs**:
- **Compute**: GPU hours ($0.50-$8/hour per GPU)
- **Storage**: Model checkpoints, datasets ($0.02-$0.20/GB/month)
- **Network**: Data transfer ($0.01-$0.12/GB)
- **Memory**: RAM if using CPU offloading ($0.001-$0.005/GB/hour)

**Hidden Costs**:
- **Failed runs**: Wasted compute from crashes/bugs
- **Hyperparameter search**: Multiple training runs
- **Development time**: Engineer time debugging/optimizing
- **Idle time**: Waiting for resources

### Cost Formula

```
Total Cost = (GPU_hours × GPU_price) +
             (Storage_GB × Storage_price × Days) +
             (Transfer_GB × Transfer_price) +
             (Failed_runs × Run_cost)
```

**Example** (7B model, 100K steps):
```
GPUs: 8× A100 (80GB)
Time: 24 hours
Price: $3.67/hour per GPU

Cost = 8 GPUs × 24 hours × $3.67/GPU/hour
     = $705.60 for single run
```

---

## Cloud Provider Pricing

### AWS Pricing (as of 2024)

**On-Demand Instances**:

| Instance | GPUs | GPU Type | vCPUs | RAM | Price/hour | Price/GPU/hour |
|----------|------|----------|-------|-----|------------|----------------|
| p4d.24xlarge | 8 | A100 (40GB) | 96 | 1152GB | $32.77 | $4.10 |
| p4de.24xlarge | 8 | A100 (80GB) | 96 | 1152GB | $40.96 | $5.12 |
| p5.48xlarge | 8 | H100 (80GB) | 192 | 2048GB | $98.32 | $12.29 |
| p3.16xlarge | 8 | V100 (16GB) | 64 | 488GB | $24.48 | $3.06 |
| g5.48xlarge | 8 | A10G (24GB) | 192 | 768GB | $16.29 | $2.04 |

**Spot Instance Discounts**: 50-90% cheaper
- A100: $1.50-$2.50/GPU/hour (vs $5.12 on-demand)
- V100: $0.90-$1.50/GPU/hour (vs $3.06 on-demand)

---

### Google Cloud Pricing (as of 2024)

**On-Demand Instances**:

| Instance | GPUs | GPU Type | vCPUs | RAM | Price/hour | Price/GPU/hour |
|----------|------|----------|-------|-----|------------|----------------|
| a2-highgpu-8g | 8 | A100 (40GB) | 96 | 680GB | $29.39 | $3.67 |
| a2-megagpu-16g | 16 | A100 (40GB) | 96 | 1360GB | $55.74 | $3.48 |
| a2-ultragpu-8g | 8 | A100 (80GB) | 96 | 1360GB | $35.73 | $4.47 |
| a3-highgpu-8g | 8 | H100 (80GB) | 208 | 1872GB | $74.16 | $9.27 |

**Preemptible Discounts**: 50-90% cheaper
- A100: $1.20-$2.00/GPU/hour (vs $3.67 on-demand)
- H100: $3.50-$5.00/GPU/hour (vs $9.27 on-demand)

---

### Azure Pricing (as of 2024)

**On-Demand Instances**:

| Instance | GPUs | GPU Type | vCPUs | RAM | Price/hour | Price/GPU/hour |
|----------|------|----------|-------|-----|------------|----------------|
| Standard_ND96asr_v4 | 8 | A100 (40GB) | 96 | 900GB | $27.20 | $3.40 |
| Standard_ND96amsr_A100_v4 | 8 | A100 (80GB) | 96 | 1900GB | $32.77 | $4.10 |
| Standard_NC96ads_A100_v4 | 4 | A100 (80GB) | 96 | 880GB | $18.15 | $4.54 |

**Spot Discounts**: 60-90% cheaper
- A100: $1.00-$1.80/GPU/hour (vs $4.10 on-demand)

---

### Lambda Labs / CoreWeave (GPU-Focused)

**Significantly Cheaper**:

| Provider | GPU | Price/hour | vs AWS |
|----------|-----|------------|--------|
| Lambda Labs | A100 (40GB) | $1.10 | 73% cheaper |
| Lambda Labs | A100 (80GB) | $1.29 | 75% cheaper |
| CoreWeave | A100 (80GB) | $2.06 | 60% cheaper |
| CoreWeave | H100 (80GB) | $4.76 | 61% cheaper |

---

## Spot Instance Strategies

### Understanding Spot Instances

**Pros**:
- 50-90% cost savings
- Same performance as on-demand
- Good availability for most GPU types

**Cons**:
- Can be interrupted (2-minute warning)
- Need checkpointing strategy
- Variable pricing

### Spot Instance Best Practices

#### 1. Implement Robust Checkpointing

```python
import os
import time

def train_with_checkpointing(model_engine, dataloader):
    """Training loop with frequent checkpoints for spot instances."""

    # Resume from latest checkpoint
    checkpoint_dir = 'checkpoints'
    start_step = 0

    if os.path.exists(f'{checkpoint_dir}/latest'):
        print("Resuming from checkpoint...")
        _, client_state = model_engine.load_checkpoint(checkpoint_dir)
        start_step = client_state.get('step', 0)

    # Training loop
    for step in range(start_step, total_steps):
        # Train step
        loss = model_engine(next(dataloader))
        model_engine.backward(loss)
        model_engine.step()

        # Checkpoint every 100 steps (adjust based on step time)
        if step % 100 == 0:
            client_state = {'step': step, 'loss': loss.item()}
            model_engine.save_checkpoint(checkpoint_dir, client_state=client_state)

            # Update 'latest' symlink
            os.system(f'cd {checkpoint_dir} && ln -sf step_{step} latest')
```

#### 2. Use Spot Fleet (Multiple Zones)

Request instances across multiple availability zones:

**AWS**:
```bash
aws ec2 request-spot-fleet \
    --spot-fleet-request-config file://spot-fleet-config.json
```

**spot-fleet-config.json**:
```json
{
  "AllocationStrategy": "lowestPrice",
  "IamFleetRole": "arn:aws:iam::...",
  "TargetCapacity": 8,
  "LaunchSpecifications": [
    {
      "InstanceType": "p4d.24xlarge",
      "SpotPrice": "20.00",
      "AvailabilityZone": "us-east-1a"
    },
    {
      "InstanceType": "p4d.24xlarge",
      "SpotPrice": "20.00",
      "AvailabilityZone": "us-east-1b"
    }
  ]
}
```

#### 3. Monitor Spot Price History

```python
import boto3

def get_spot_price_history(instance_type, days=7):
    """Get spot price history to choose best time/zone."""
    ec2 = boto3.client('ec2')

    response = ec2.describe_spot_price_history(
        InstanceTypes=[instance_type],
        ProductDescriptions=['Linux/UNIX'],
        StartTime=datetime.now() - timedelta(days=days)
    )

    prices = response['SpotPriceHistory']
    avg_price = sum(float(p['SpotPrice']) for p in prices) / len(prices)

    print(f"Average spot price: ${avg_price:.2f}/hour")
    print(f"Min: ${min(float(p['SpotPrice']) for p in prices):.2f}")
    print(f"Max: ${max(float(p['SpotPrice']) for p in prices):.2f}")

    return avg_price
```

#### 4. Set Maximum Price

```bash
# AWS spot instance with max price
aws ec2 run-instances \
    --instance-type p4d.24xlarge \
    --spot-instance-type one-time \
    --spot-price "25.00"  # Max price willing to pay
```

### Handling Spot Interruptions

**2-Minute Warning Handler**:
```python
import requests
import threading
import time

def check_spot_termination():
    """Check for spot termination notice."""
    try:
        response = requests.get(
            'http://169.254.169.254/latest/meta-data/spot/termination-time',
            timeout=1
        )
        if response.status_code == 200:
            return True
    except:
        pass
    return False

def termination_handler(model_engine):
    """Monitor and handle spot termination."""
    while True:
        if check_spot_termination():
            print("SPOT TERMINATION NOTICE! Saving checkpoint...")
            model_engine.save_checkpoint('emergency', tag='spot_termination')
            print("Checkpoint saved. Exiting gracefully.")
            exit(0)
        time.sleep(5)

# Start monitoring thread
thread = threading.Thread(target=termination_handler, args=(model_engine,))
thread.daemon = True
thread.start()
```

---

## Configuration Cost Tradeoffs

### ZeRO Stage vs Cost

| ZeRO Stage | Speed | GPU Needed | Training Time | Cost (8× A100) |
|------------|-------|------------|---------------|----------------|
| Stage 0 | Fastest (1.0×) | Most (76GB) | 24h | $983 |
| Stage 1 | Fast (0.95×) | Less (62GB) | 25h | $1,033 |
| Stage 2 | Moderate (0.85×) | Moderate (55GB) | 28h | $1,157 |
| Stage 3 | Slower (0.70×) | Least (48GB) | 34h | $1,405 |

**Recommendation**: Use highest ZeRO stage that fits in memory
- If model fits with Stage 0 → Use Stage 0 (cheapest)
- If requires Stage 3 → Use Stage 3 (enables training, worth extra cost)

---

### Offloading vs Adding GPUs

**Scenario**: 13B model, 8× A100 (80GB)

**Option A: CPU Offload (ZeRO-3)**
- GPUs: 8× A100
- Cost: $40.96/hour
- Time: 48 hours (slower due to offload)
- **Total: $1,966**

**Option B: More GPUs (ZeRO-2, no offload)**
- GPUs: 16× A100
- Cost: $81.92/hour
- Time: 28 hours (faster, no offload)
- **Total: $2,294**

**Verdict**: CPU offload cheaper by $328 (14% savings)

---

### Batch Size Impact

Larger batch sizes = faster training but same cost if fitting in same GPUs:

| Batch Size | Time per Step | Steps Needed | Total Time | Cost |
|------------|---------------|--------------|------------|------|
| 8 | 500ms | 100,000 | 14h | $578 |
| 16 | 550ms | 50,000 | 8h | $330 |
| 32 | 600ms | 25,000 | 4h | $165 |

**Recommendation**: Use largest batch size that fits in memory

---

## Memory vs Speed Optimization

### Decision Matrix

```
                    ┌─────────────────────────┐
                    │   Model Fits in GPU?    │
                    └────────┬────────────────┘
                             │
                    ┌────────┴────────┐
                   Yes               No
                    │                 │
                    ▼                 ▼
           ┌────────────────┐  ┌──────────────┐
           │  Optimize for  │  │ Must use     │
           │  Speed         │  │ ZeRO-3 or    │
           │                │  │ Offloading   │
           │ - ZeRO-0/1     │  │              │
           │ - Large batch  │  │ - ZeRO-3     │
           │ - No offload   │  │ - CPU offload│
           │                │  │ - Add GPUs   │
           └────────────────┘  └──────────────┘
                    │                 │
                    ▼                 ▼
             ┌─────────────────────────────┐
             │  Minimize Cost:             │
             │  - Use spot instances       │
             │  - Checkpoint frequently    │
             │  - Monitor and optimize     │
             └─────────────────────────────┘
```

---

## Multi-Node Cost Considerations

### Single Node vs Multi-Node

**7B Model Training (100K steps)**:

| Setup | GPUs | Time | Cost/hour | Total Cost | Notes |
|-------|------|------|-----------|------------|-------|
| 1 Node (8 GPUs) | 8 | 24h | $40.96 | $983 | Baseline |
| 2 Nodes (16 GPUs) | 16 | 14h | $81.92 | $1,147 | 17% more expensive |
| 4 Nodes (32 GPUs) | 32 | 9h | $163.84 | $1,475 | 50% more expensive |

**Why more expensive?**:
- Communication overhead reduces efficiency
- Not perfectly linear scaling

**When multi-node worth it**:
- Model doesn't fit in single node
- Time-critical (deadline)
- Research iteration speed matters

---

### Network Costs

**Data Transfer Pricing**:
- **Intra-region**: Free (within same region)
- **Inter-region**: $0.02/GB (cross-region)
- **Internet egress**: $0.09/GB (out to internet)

**Minimize transfer costs**:
- Use same region for all nodes
- Store checkpoints in cloud storage (S3, GCS)
- Download datasets once to shared storage

---

## Cost Calculation Examples

### Example 1: LLaMA-7B Fine-Tuning

**Setup**:
- Model: LLaMA-7B
- Dataset: 50K examples
- Training steps: 10,000
- Hardware: 8× A100 (80GB)

**Configuration**:
```json
{
  "zero_optimization": {"stage": 2},
  "fp16": {"enabled": true},
  "train_micro_batch_size_per_gpu": 4
}
```

**Cost Breakdown**:
```
Provider: AWS (spot instance)
Instance: p4de.24xlarge (8× A100 80GB)
Spot price: $15.00/hour
Training time: 6 hours

Compute: 6h × $15.00/h = $90
Storage: 50GB × $0.023/GB/month × (6h/720h) = $0.01
Transfer: Negligible (same region)

Total: ~$90
```

---

### Example 2: GPT-3 13B Pre-Training

**Setup**:
- Model: GPT-3 13B
- Dataset: 300B tokens
- Training steps: 300,000
- Hardware: 32× A100 (80GB) - 4 nodes

**Configuration**:
```json
{
  "zero_optimization": {"stage": 3},
  "bf16": {"enabled": true},
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 8
}
```

**Cost Breakdown**:
```
Provider: Lambda Labs
Instance: 4× 8-GPU nodes (32× A100 80GB)
Price: 32 GPUs × $1.29/GPU/hour = $41.28/hour
Training time: 480 hours (20 days)

Compute: 480h × $41.28/h = $19,814
Storage: 500GB × $0.10/GB/month × 1 month = $50
Transfer: Minimal

Total: ~$19,864
```

**Comparison with AWS on-demand**:
- AWS: 480h × ($40.96 × 4 nodes) = $78,604
- **Savings: $58,740 (75% cheaper!)**

---

### Example 3: Hyperparameter Search

**Setup**:
- Model: BERT-Large
- Configurations to test: 20
- Training time per config: 2 hours
- Hardware: 8× V100

**Cost Calculation**:
```
Provider: GCP (preemptible)
Instance: a2-highgpu-8g (8× A100)
Preemptible price: $12.00/hour
Time: 20 configs × 2h = 40 hours

Compute: 40h × $12.00/h = $480

With parallel runs (4 instances):
Compute: 4 instances × 10h × ($12.00 × 4) = $1,920
BUT completes in 10 hours vs 40 hours

Time value: Worth it if 30 hours saves > $1,440 in labor
```

---

## Cost Reduction Strategies

### 1. Use Gradient Checkpointing

**Memory savings**: 40-60%
**Speed impact**: 20-33% slower
**Cost impact**: Can use fewer/smaller GPUs

**Example**:
```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Or in DeepSpeed config
{
  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true
  }
}
```

**Savings**:
- Before: Need 8× A100 (80GB) = $40.96/hour
- After: Need 8× A100 (40GB) = $32.77/hour
- **Save: $8.19/hour (20%)**

---

### 2. Mix Spot and On-Demand

Use on-demand for critical jobs, spot for experiments:

```
Critical training: On-demand (reliable)
Hyperparameter search: Spot (interruptible OK)
Checkpointed long runs: Spot (resumable)
```

**Cost allocation**:
```
80% of budget on spot (50-70% savings)
20% of budget on on-demand (reliability)
Overall savings: 40-56%
```

---

### 3. Compress Gradients for Multi-Node

**1-bit Adam** reduces inter-node traffic by 32×:

```json
{
  "optimizer": {
    "type": "OneBitAdam",
    "params": {
      "lr": 1e-4,
      "freeze_step": 1000
    }
  }
}
```

**Savings**:
- Multi-node (4 nodes): 2.4× faster
- Time: 20h → 8.3h
- **Cost savings: $475 (58%)**

---

### 4. Optimize Data Loading

Slow data loading = idle GPUs = wasted money:

```python
dataloader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=8,  # Increase workers
    pin_memory=True,  # Fast GPU transfer
    prefetch_factor=2,  # Prefetch batches
    persistent_workers=True  # Keep workers alive
)
```

**Impact**:
- Before: 10% of time waiting for data
- After: < 1% waiting
- **Savings: 9% of compute cost**

---

### 5. Schedule Training During Off-Peak

Some clouds have time-of-day pricing:

```
Peak hours (9AM-5PM): $5.12/GPU/hour
Off-peak (9PM-5AM): $3.84/GPU/hour
Weekend: $3.20/GPU/hour
```

**Savings**: 25-38% by running nights/weekends

---

## ROI Optimization

### Calculate Value of Speed

**Formula**:
```
Value of Time = (Engineer_hourly_rate × Time_saved) - Extra_compute_cost
```

**Example**:
```
Engineer rate: $100/hour
Option A: 40 hours training, $500 compute
Option B: 10 hours training, $800 compute

Time saved: 30 hours
Value: (30h × $100/h) - $300 extra compute
     = $3,000 - $300
     = $2,700 net benefit

ROI: 900% return on extra compute investment
```

**Conclusion**: Spending more on compute often worth it for faster iteration.

---

### Optimize for Iteration Speed

**Research scenarios**: Fast iteration > cost savings

```
Scenario A: $100/run, 4 hours
  - Can try 6 ideas per day
  - Daily cost: $600
  - Experiments: 6

Scenario B: $50/run, 12 hours
  - Can try 2 ideas per day
  - Daily cost: $100
  - Experiments: 2

Value: 3× more experiments worth 6× higher daily cost
```

**Production scenarios**: Cost savings > speed

```
Final training run: Don't need speed
  - Use spot instances
  - Use cheaper GPUs (V100 vs A100)
  - Optimize for cost, not time
```

---

## Cost Optimization Checklist

- [ ] **Use spot instances** for all interruptible workloads
- [ ] **Implement checkpointing** every 10-30 minutes
- [ ] **Choose right ZeRO stage** (highest that fits)
- [ ] **Optimize batch size** (largest that fits)
- [ ] **Enable gradient checkpointing** if memory-constrained
- [ ] **Use compression** for multi-node (1-bit Adam)
- [ ] **Monitor GPU utilization** (should be >80%)
- [ ] **Optimize data loading** (num_workers, prefetch)
- [ ] **Use cheaper providers** (Lambda, CoreWeave vs AWS)
- [ ] **Schedule off-peak** when possible
- [ ] **Clean up resources** (stop instances, delete checkpoints)
- [ ] **Track costs** with cloud billing alerts

---

## Cost Tracking Tools

### AWS Cost Explorer

```bash
# Get costs for last 30 days
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-31 \
    --granularity DAILY \
    --metrics BlendedCost
```

### GCP Billing Reports

```bash
# Export billing to BigQuery
bq query --use_legacy_sql=false '
SELECT
  service.description,
  SUM(cost) as total_cost
FROM
  `project.dataset.gcp_billing_export`
WHERE
  DATE(usage_start_time) >= "2024-01-01"
GROUP BY
  service.description
ORDER BY
  total_cost DESC
'
```

### Set Budget Alerts

**AWS Budget**:
```bash
aws budgets create-budget \
    --account-id 123456789 \
    --budget file://budget.json
```

**budget.json**:
```json
{
  "BudgetName": "Monthly GPU Budget",
  "BudgetLimit": {
    "Amount": "5000",
    "Unit": "USD"
  },
  "TimeUnit": "MONTHLY",
  "BudgetType": "COST"
}
```

---

## Summary: Cost Optimization Decision Tree

```
┌─────────────────────────────────────┐
│ What's your priority?               │
└────────┬────────────────────────────┘
         │
    ┌────┴────┐
    │         │
  Speed    Cost
    │         │
    ▼         ▼
┌───────┐   ┌──────────┐
│- More │   │- Spot    │
│  GPUs │   │- ZeRO-3  │
│- Fast │   │- Compress│
│  GPUs │   │- Cheap   │
│- Multi│   │  provider│
│  node │   └──────────┘
└───────┘
```

**Key Takeaway**: There's no one-size-fits-all. Optimize based on your specific constraints (budget, time, iteration speed).

---

## Additional Resources

- **[AWS EC2 Pricing](https://aws.amazon.com/ec2/pricing/)** - Latest AWS GPU pricing
- **[GCP Pricing Calculator](https://cloud.google.com/products/calculator)** - GCP cost estimator
- **[Lambda Labs Pricing](https://lambdalabs.com/service/gpu-cloud)** - GPU cloud pricing
- **[Cost Calculator Tool](../tools/cost_calculator.py)** - Automated cost calculations

**Happy cost-optimized training!** 💰
