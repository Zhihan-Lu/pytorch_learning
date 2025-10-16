# TensorBoard + PyTorch Profiler Integration Guide

## 🎯 Overview

TensorBoard provides an interactive web interface to visualize PyTorch Profiler traces. It transforms raw profiling data into intuitive visualizations that help you identify performance bottlenecks.

## 🚀 Starting TensorBoard

### Basic Usage
```bash
# Start TensorBoard pointing to your profiler directory
tensorboard --logdir ./profiler/ml-1m-l200/

# Or start with specific port
tensorboard --logdir ./profiler/ml-1m-l200/ --port 6006
```

### Access the Interface
Open your browser and go to: `http://localhost:6006`

## 📊 TensorBoard Profiler Views

### 1. **Overview Tab**
**Purpose**: High-level performance summary
**What you'll see**:
- Total training time breakdown
- GPU/CPU utilization percentages
- Memory usage patterns
- Key performance metrics

**Key Metrics**:
- **GPU Utilization**: How much of your GPU is being used
- **CPU Utilization**: CPU usage during training
- **Memory Usage**: Peak and average memory consumption
- **Step Time**: Time per training step

### 2. **Operator View**
**Purpose**: Detailed operator-level performance analysis
**What you'll see**:
- List of all operations sorted by execution time
- Self time vs total time for each operation
- Memory usage per operation
- Call count and frequency

**Key Columns**:
- **Self Duration**: Time spent in the operation itself
- **Total Duration**: Time including sub-operations
- **Memory**: Memory allocated by the operation
- **Count**: Number of times the operation was called

### 3. **Kernel View** (GPU only)
**Purpose**: GPU kernel execution details
**What you'll see**:
- Individual GPU kernel execution times
- Kernel efficiency metrics
- Memory bandwidth utilization
- Kernel launch overhead

**Important Metrics**:
- **Kernel Duration**: Time spent in GPU kernels
- **Memory Bandwidth**: Data transfer efficiency
- **Compute Efficiency**: How well kernels utilize GPU cores

### 4. **Trace View**
**Purpose**: Timeline visualization of operations
**What you'll see**:
- Chronological timeline of all operations
- Operation dependencies and overlaps
- Memory allocation/deallocation events
- GPU/CPU activity patterns

**How to Use**:
- **Zoom**: Mouse wheel or pinch to zoom in/out
- **Pan**: Click and drag to move around timeline
- **Select**: Click on operations to see details
- **Filter**: Use search box to find specific operations

### 5. **Memory View**
**Purpose**: Memory allocation and usage analysis
**What you'll see**:
- Memory allocation timeline
- Peak memory usage
- Memory fragmentation
- Allocation patterns by operation

**Key Insights**:
- **Memory Growth**: Identify memory leaks
- **Peak Usage**: Find memory bottlenecks
- **Fragmentation**: Understand memory efficiency

## 🔍 How to Analyze Your Model

### Step 1: Start with Overview
1. Look at **GPU Utilization** - should be >80% for efficient training
2. Check **Step Time** - identify if training is slow
3. Examine **Memory Usage** - look for spikes or leaks

### Step 2: Dive into Operator View
1. Sort by **Self Duration** to find slowest operations
2. Look for operations taking >10% of total time
3. Check **Memory** column for memory-intensive operations

### Step 3: Use Trace View for Timeline Analysis
1. Look for **gaps** in GPU utilization (idle time)
2. Identify **overlaps** between CPU and GPU operations
3. Find **bottlenecks** where operations are queued

### Step 4: Analyze Memory Patterns
1. Check for **memory spikes** during specific operations
2. Look for **gradual memory growth** (potential leaks)
3. Identify **memory fragmentation** issues

## 🎯 Common Bottlenecks to Look For

### 1. **Data Loading Bottlenecks**
**Signs**:
- Low GPU utilization (<50%)
- Long gaps in GPU timeline
- High CPU usage during data loading

**In Trace View**: Look for long `DataLoader` operations

### 2. **Memory Transfer Issues**
**Signs**:
- Frequent CPU↔GPU transfers
- High memory bandwidth usage
- Overlapping memory operations

**In Operator View**: Look for `to()`, `cuda()`, `cpu()` operations

### 3. **Attention Computation Bottlenecks**
**Signs**:
- High self-duration for attention operations
- Memory spikes during attention
- Poor GPU utilization in attention kernels

**In Kernel View**: Look for attention-related kernels

### 4. **Loss Computation Issues**
**Signs**:
- High time spent in loss functions
- Memory allocation during loss computation
- Many small operations in loss

**In Operator View**: Look for `SampledSoftmaxLoss` operations

## 🛠️ Practical Analysis Workflow

### 1. **Quick Performance Check**
```bash
# Start TensorBoard
tensorboard --logdir ./profiler/ml-1m-l200/

# Open browser to http://localhost:6006
# Go to PROFILER tab
# Check Overview for GPU utilization
```

### 2. **Detailed Bottleneck Analysis**
1. **Overview Tab**: Check overall performance
2. **Operator View**: Sort by Self Duration
3. **Trace View**: Look for timeline patterns
4. **Memory View**: Check memory usage

### 3. **Compare Different Runs**
```bash
# Run with different configurations
python main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-profiling.gin

# TensorBoard will show multiple runs for comparison
```

## 📈 Interpreting Results for Your Model

### For HSTU Model Specifically:

**Expected Patterns**:
- **Forward Pass**: Should dominate training time
- **Attention Layers**: May show high memory usage
- **Loss Computation**: Sampled softmax with 128 negatives
- **Embedding Lookups**: Should be fast with small vocab

**Red Flags**:
- **Low GPU Utilization**: Data loading or CPU bottlenecks
- **High Memory Spikes**: Inefficient memory usage
- **Long Idle Periods**: Synchronization issues
- **Frequent Memory Transfers**: Inefficient data movement

### Optimization Recommendations:

1. **If GPU Utilization < 80%**:
   - Increase `num_workers` in data loader
   - Reduce `prefetch_factor`
   - Check for CPU bottlenecks

2. **If Memory Usage is High**:
   - Reduce batch size
   - Enable gradient checkpointing
   - Use mixed precision training

3. **If Attention is Slow**:
   - Reduce sequence length
   - Use fewer attention heads
   - Consider attention optimizations

## 🔧 Advanced TensorBoard Features

### 1. **Custom Profiling Regions**
```python
# In your code, add custom regions
with record_function("my_custom_operation"):
    # Your code here
    pass
```

### 2. **Memory Profiling**
```python
# Enable detailed memory profiling
profiler = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_stack=True
)
```

### 3. **Exporting Traces**
```python
# Export to Chrome tracing format
profiler.export_chrome_trace("custom_trace.json")
```

## 🎯 Quick Start Commands

```bash
# 1. Run profiling
python main.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-profiling.gin

# 2. Start TensorBoard
tensorboard --logdir ./profiler/ml-1m-l200/

# 3. Open browser
# Go to http://localhost:6006
# Click on PROFILER tab

# 4. Analyze results
# Start with Overview → Operator View → Trace View
```

## 💡 Pro Tips

1. **Use Multiple Runs**: Compare different configurations
2. **Focus on Self Duration**: Operations taking >5% of time
3. **Check Memory Patterns**: Look for leaks or spikes
4. **GPU Utilization**: Should be consistently high
5. **Timeline Analysis**: Look for gaps and overlaps

The TensorBoard Profiler interface makes it easy to identify exactly where your model spends time and memory, helping you optimize performance effectively!
