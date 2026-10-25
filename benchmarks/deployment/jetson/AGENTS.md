# Jetson Deployment Guide

This is a human-readable and agent-readable procedure for preparing the locked
PointRAFT deployment environment on the Jetson. It does not use `sudo`, does not
modify system Python, and does not compile Open3D.

## Verified Target

```text
Device: NVIDIA Orin NX Developer Kit, 16 GB
JetPack: 5.1.3
L4T: R35.5.0
Ubuntu: 20.04 ARM64
CUDA: 11.4
Python: /usr/bin/python3, version 3.8.10
Repository: /home/crest/pointraft-paper-full
```

The system Python is used only as the base interpreter. All Python packages,
native extensions, downloaded headers, and private libraries remain under
`benchmarks/deployment/jetson/`.

## Connect

From another machine:

```bash
ssh crest@192.168.250.45
cd /home/crest/pointraft-paper-full/benchmarks/deployment/jetson
```

Run every remaining command from this directory. Relative paths in
`.env.jetson` depend on that working directory.

## Install uv

The locked environment expects uv 0.11.30:

```bash
curl -LsSf https://astral.sh/uv/0.11.30/install.sh | \
    env UV_INSTALL_DIR="${HOME}/.local/bin" UV_NO_MODIFY_PATH=1 sh
${HOME}/.local/bin/uv --version
```

## Extract Private Runtime Files

NVIDIA PyTorch requires OpenBLAS at runtime, while the two native extensions
need Python 3.8 development headers. Download and extract the Ubuntu packages
locally without installing them system-wide:

```bash
mkdir -p .runtime/openblas .runtime/python-dev .build

(
    cd .runtime/openblas
    apt-get download libopenblas0-pthread
    for archive in ./*.deb; do
        dpkg-deb --extract "${archive}" root
    done
)

(
    cd .runtime/python-dev
    apt-get download python3.8-dev libpython3.8-dev
    for archive in ./*.deb; do
        dpkg-deb --extract "${archive}" root
    done
)
```

## Create the Locked Environment

Sync all pure-Python packages and the NVIDIA ARM64 PyTorch wheel. The two native
extensions are installed separately in the next step:

```bash
${HOME}/.local/bin/uv sync \
    --locked \
    --inexact \
    --python /usr/bin/python3 \
    --no-python-downloads \
    --no-install-package torch-cluster \
    --no-install-package torch-fpsample
```

This command installs the prebuilt Open3D 0.18.0 ARM64 wheel. Do not run
`pip install open3d` without the lock because Open3D 0.19 has no compatible
Linux ARM64 wheel.

## Build the Native Extensions

Clone the exact source revisions:

```bash
git clone https://github.com/rusty1s/pytorch_cluster.git .build/pytorch_cluster
git -C .build/pytorch_cluster checkout --detach \
    29cd22bf1a5b82fc06b108d6573f81302c5d6b12

git clone https://github.com/leonardodalinky/pytorch_fpsample.git \
    .build/pytorch_fpsample
git -C .build/pytorch_fpsample checkout --detach \
    be93a0859d196d0d2d0562bfc5e44e24b01a98de
git -C .build/pytorch_fpsample apply --unidiff-zero \
    "${PWD}/patches/torch-fpsample-torch20.patch"
```

Set the private runtime and Jetson build variables:

```bash
export PRIVATE_LIB_DIR="${PWD}/.runtime/openblas/root/usr/lib/aarch64-linux-gnu/openblas-pthread"
export PYTHON_INCLUDE_ROOT="${PWD}/.runtime/python-dev/root/usr/include"
export LD_PRELOAD="${PRIVATE_LIB_DIR}/libopenblas.so.0"
export LD_LIBRARY_PATH="${PRIVATE_LIB_DIR}"
export CPLUS_INCLUDE_PATH="${PYTHON_INCLUDE_ROOT}:${PYTHON_INCLUDE_ROOT}/python3.8:${PYTHON_INCLUDE_ROOT}/aarch64-linux-gnu/python3.8"
export CUDA_HOME=/usr/local/cuda
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST=8.6+PTX
```

JetPack's PyTorch 2.0 extension helper does not list Orin `sm_87`. The compute
8.6 PTX fallback is JIT-compiled for `sm_87` when the extension loads.

Build and install both extensions into `.venv`:

```bash
${HOME}/.local/bin/uv pip install \
    --python .venv/bin/python \
    --no-build-isolation \
    --no-deps \
    .build/pytorch_cluster

${HOME}/.local/bin/uv pip install \
    --python .venv/bin/python \
    --no-build-isolation \
    --no-deps \
    .build/pytorch_fpsample
```

## Verify

Confirm package loading and inspect the repository's canonical checkpoint:

```bash
${HOME}/.local/bin/uv run --frozen --no-sync --env-file .env.jetson \
    python -c 'import open3d, torch, torch_cluster, torch_fpsample; print(open3d.__version__, torch.__version__, torch.cuda.is_available())'

sha256sum ../../../weights/pointraft.pth
```

Run the shared deployment check:

```bash
${HOME}/.local/bin/uv run --frozen --no-sync --env-file .env.jetson \
    python -m benchmarks deployment smoke --device cuda
```

The pipeline records the actual checkpoint SHA-256 in each run's `system.json`.
For formal 10 W, 15 W, and 25 W benchmark commands, see the deployment section
in [`../../README.md`](../../README.md). Power-mode switching is a separate manual
system operation and is not performed by these installation commands.

## Existing Installations

Installations created before the Jetson project moved into this directory keep
their generated `.venv`, `.runtime`, and `.build` directories one level above.
Do not move the old virtual environment because its scripts contain absolute
paths. Keep it as rollback evidence while repeating this guide here, then remove
the old generated directories only after the new smoke test succeeds.

## Rebuild

For a clean rebuild, remove only `.venv`, `.runtime`, and `.build` from this
directory, then repeat this guide. These paths are ignored by Git. Do not remove
the repository-level `data/`, `weights/`, or `results/` directories.
