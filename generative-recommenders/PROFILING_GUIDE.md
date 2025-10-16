# PyTorch Profiler Integration for Generative Recommenders

This guide explains how to use PyTorch Profiler to identify performance bottlenecks in your model training.

## Quick Start

### 1. Run Training with Profiling

```bash
# Create necessary directories
mkdir -p logs/ml-1m-l200/
mkdir -p profiler/ml-1m-l200/

# Run training with profiling enabled
CUDA_VISIBLE_DEVICES=0 python3 main.py \
    --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-profiling.gin \
    --master_port=12345 \
    2>&1 | tee logs/ml-1m-l200/hstu-sampled-softmax-n128-profiling.log
```

### 2. View Results in TensorBoard

```bash
# Start TensorBoard to view profiling traces
tensorboard --logdir ./profiler/ml-1m-l200/
```

Then open your browser to `http://localhost:6006` and navigate to the "PROFILER" tab.

### 3. Analyze Results Programmatically

```bash
# Run the analysis script
python analyze_profiler.py --profiler_dir ./profiler/ml-1m-l200/your_model_directory/
```

## Profiling Configuration

The profiling is controlled by these parameters in the gin config:

```gin
# Enable/disable profiling
train_fn.enable_profiling = True

# Profiler schedule (steps)
train_fn.profile_warmup_steps = 5    # Steps to warm up before profiling
train_fn.profile_active_steps = 10   # Steps to actively profile
train_fn.profile_repeat_steps = 1    # Number of profiling cycles
```

## What Gets Profiled

The profiler captures detailed information about:

1. **Forward Pass** (`forward_pass`): Model inference including embeddings and attention
2. **Loss Computation** (`loss_computation`): Sampled softmax loss calculation
3. **Backward Pass** (`backward_pass`): Gradient computation
4. **Optimizer Step** (`optimizer_step`): Parameter updates

## Understanding the Results

### Key Metrics to Look For

1. **GPU Utilization**: Look for gaps in GPU usage
2. **Memory Usage**: Check for memory spikes or inefficient allocation
3. **Operator Timing**: Identify slowest operations
4. **Data Loading**: Check if data loading is a bottleneck

### Common Bottlenecks

1. **Data Loading**: If `DataLoader` operations take significant time
2. **Memory Transfers**: CPU-GPU data movement
3. **Attention Computation**: Large sequence lengths can be expensive
4. **Loss Computation**: Sampled softmax with many negatives
5. **Embedding Lookups**: Large vocabulary sizes

### TensorBoard Profiler Views

- **Overview**: High-level performance summary
- **Operator View**: Detailed operator-level timing
- **Kernel View**: GPU kernel execution details
- **Trace View**: Timeline of operations
- **Memory View**: Memory allocation patterns

## Performance Optimization Tips

Based on profiling results, consider these optimizations:

### 1. Data Loading
```python
# Increase num_workers if CPU-bound
create_data_loader.num_workers = 16

# Adjust prefetch factor
create_data_loader.prefetch_factor = 4
```

### 2. Model Optimization
```python
# Use mixed precision training
train_fn.main_module_bf16 = True

# Enable TF32 for faster training
train_fn.enable_tf32 = True
```

### 3. Loss Computation
```python
# Reduce number of negatives for faster training
train_fn.num_negatives = 64

# Use activation checkpointing for memory efficiency
train_fn.loss_activation_checkpoint = True
```

### 4. Batch Size Optimization
```python
# Increase batch size if memory allows
train_fn.local_batch_size = 256
```

## Troubleshooting

### Profiler Not Starting
- Ensure `enable_profiling = True` in gin config
- Check that you have sufficient disk space for traces
- Verify PyTorch version supports profiler

### High Memory Usage
- Reduce `profile_active_steps` to limit memory overhead
- Use `profile_memory = False` in profiler config if needed

### Slow Profiling
- Reduce `profile_active_steps` and `profile_repeat_steps`
- Profile only specific epochs by modifying the training loop

## Example Analysis Output

```
=== Performance Analysis ===
Event Name                               Count    Avg (ms)     Max (ms)     Total (ms)  
------------------------------------------------------------------------------------------
forward_pass                            10       45.23        52.11        452.30      
loss_computation                        10       23.45        28.33        234.50      
backward_pass                           10       67.89        75.22        678.90      
optimizer_step                          10       12.34        15.67        123.40      

=== Potential Bottlenecks ===
- backward_pass: 67.89ms average
- forward_pass: 45.23ms average

=== Recommendations ===
1. Use TensorBoard to visualize the trace
2. Look for long-running operations in forward/backward pass
3. Check GPU utilization gaps
4. Consider reducing model complexity or batch size
```

## Advanced Usage

### Custom Profiling Regions

You can add custom profiling regions in your code:

```python
with record_function("custom_operation"):
    # Your custom code here
    pass
```

### Memory Profiling

Enable detailed memory profiling:

```python
profiler = profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_stack=True
)
```

### Exporting Traces

Traces are automatically saved to TensorBoard format. You can also export to Chrome tracing format:

```python
profiler.export_chrome_trace("trace.json")
```
