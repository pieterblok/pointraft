# Pascal Deployment Environment

This isolated x86 environment supports NVIDIA Pascal GPUs such as the GTX 1080
with CUDA capability `sm_61`. The repository-level CUDA 12.8 PyTorch wheel
starts at `sm_70`, so it cannot execute kernels on these GPUs.

Create the locked CUDA 12.6 environment from the repository root:

```bash
uv sync --project benchmarks/deployment/pascal --locked
```

Run the shared deployment pipeline from the repository root:

```bash
uv run --project benchmarks/deployment/pascal --frozen --no-sync \
  python -m benchmarks deployment smoke --device cuda
uv run --project benchmarks/deployment/pascal --frozen --no-sync \
  python -m benchmarks deployment run --device cuda --limit 10 --warmup 3
uv run --project benchmarks/deployment/pascal --frozen --no-sync \
  python -m benchmarks deployment run --device cuda
```

The environment remains under `benchmarks/deployment/pascal/.venv`. Benchmark
code, data, checkpoints, and output paths are shared with every other platform.
