# Deployment Environment

The deployment benchmark uses one shared implementation on x86 workstations and
Jetson. Only the Python environments differ.

## Modern x86 Linux

Use the repository-level Python 3.12/CUDA 12.8 environment. From the repository
root:

```bash
uv sync --locked
```

This installs the Open3D 0.19, OpenCV, PyTorch, PyG, and FPS dependencies needed
by the deployment pipeline.

## Windows x86-64

Windows with a modern NVIDIA GPU uses the repository-level Python 3.12/CUDA
12.8 lock. Install Visual Studio Build Tools 2022 with the C++ workload because
`torch-fpsample` is compiled from source. Then run the initial sync from an x64
Developer Command Prompt:

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
set DISTUTILS_USE_SDK=1
set CL=/Zc:__cplusplus
set MAX_JOBS=1
uv sync --all-groups
```

The compiled wheel is retained in uv's cache, so later syncs do not normally
need to rebuild it. Windows and modern Linux use the same cu128 dependency
versions and shared deployment commands.

## NVIDIA Pascal x86 Linux

Pascal GPUs such as the GTX 1080 require the isolated CUDA 12.6 environment in
[`pascal/`](pascal/), because the repository-level CUDA 12.8 PyTorch wheel does
not include `sm_61` kernels. The subproject changes only the dependency stack;
it runs the same deployment implementation, dataset, checkpoint, and result
schema as every other platform.

## Jetson

JetPack 5.1 requires the locked Python 3.8 ARM64 environment in
[`jetson/`](jetson/). Open3D 0.18 is installed from its prebuilt ARM64 wheel;
Open3D source compilation and system-wide installation are not required.

See [`jetson/AGENTS.md`](jetson/AGENTS.md) for the complete SSH and no-sudo
manual installation procedure. The guide is written for both people and
automation agents.
