# Deployment Environment

The deployment benchmark uses one shared implementation on x86 workstations and
Jetson. Only the Python environments differ.

## x86 Linux

Use the repository-level Python 3.12/CUDA 12.8 environment. From the repository
root:

```bash
uv sync --locked
```

This installs the Open3D 0.19, OpenCV, PyTorch, PyG, and FPS dependencies needed
by the deployment pipeline.

## Jetson

JetPack 5.1 requires the locked Python 3.8 ARM64 environment in this directory.
Open3D 0.18 is installed from its prebuilt ARM64 wheel; Open3D source compilation
and system-wide installation are not required.

See [`AGENTS.md`](AGENTS.md) for the complete SSH and no-sudo manual installation
procedure. The guide is written for both people and automation agents.
