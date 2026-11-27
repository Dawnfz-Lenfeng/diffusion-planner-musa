# Diffusion Planner x nuPlan mini baseline — MUSA image (OPTIONAL).
#
# Primary path is source-install on an existing MUSA environment
# (docs/REPRODUCE.md + docker/requirements-lock.txt); the MUSA base image is
# environment-provided. This Dockerfile is a convenience recipe.
#
# This is the MUSA adaptation delivery: the pipeline runs on Moore Threads
# GPUs via torch_musa + MCCL. The MUSA toolchain comes from the training-suite
# base image — fill in the FROM line with your own registry/image tag, then build.
# No pip torch here: the base image's torch/torch_musa must never be shadowed.
#
# Dependencies install into an overlay dir (/workspace/pylibs) activated via
# PYTHONPATH — same layout as docs/REPRODUCE.md (which mirrors this file for
# non-docker environments).
#
# A secondary CUDA reference image lives in ./Dockerfile.cuda (upstream-native
# environment, for comparison runs only).

# FROM <client-registry>/<training-suite-image>:<tag>

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git unzip wget ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN git clone https://github.com/motional/nuplan-devkit.git third_party/nuplan-devkit \
 && cd third_party/nuplan-devkit \
 && git checkout e9241677997dd86bfc0bcd44817ab04fe631405b

COPY . /workspace
COPY scripts/requirements-nuplan-x86.txt docker/requirements-lock.txt /tmp/
# pip 24 overlay: newer pip rejects old package metadata (see docs/debug-log.md)
RUN pip install --no-cache-dir --target=/workspace/pylibs pip==24.0
ENV PIP="env PYTHONPATH=/workspace/pylibs python3 -m pip"
RUN $PIP install --no-cache-dir --target=/workspace/pylibs --upgrade \
        -r /tmp/requirements-lock.txt \
 && $PIP install --no-cache-dir --target=/workspace/pylibs --no-deps \
        pytorch-lightning==2.0.1 timm==1.0.10 mmengine tensorboard==2.11.2 \
 && cp -r /workspace/diffusion_planner /workspace/pylibs/

ENV PYTHONPATH=/workspace/pylibs:/workspace:/workspace/third_party/nuplan-devkit \
    NUPLAN_DATA_ROOT=/data/sets/nuplan \
    NUPLAN_MAPS_ROOT=/data/sets/nuplan/maps \
    NUPLAN_EXP_ROOT=/workspace/exp \
    MCCL_IB_DISABLE=1 \
    PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True \
    OMP_NUM_THREADS=8

# training:
#   python -m torch.distributed.run --nnodes 1 --nproc-per-node 8 --standalone \
#       train_predictor.py --device musa --batch_size 2048 --train_epochs 500 \
#       --train_set <npz_dir> --train_set_list <npz_dir>/diffusion_planner_training.json
CMD ["/bin/bash"]
