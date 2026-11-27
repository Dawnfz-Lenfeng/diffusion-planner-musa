# Docker（可选）

主路径是 **源码安装到已有 MUSA 环境**（`docs/REPRODUCE.md` + `docker/requirements-lock.txt`），
MUSA 基础镜像由你的环境提供，适配代码装在你现有环境之上。
本目录的两份 Dockerfile 只是额外便利：MUSA 版需你填入自己的基础镜像；CUDA 参照版可在任意 x86 docker 机器构建。
镜像内安装的依赖与主路径同为 `docker/requirements-lock.txt`（213 包精确版本），行为一致。

| 文件 | 角色 | 基础镜像 |
|---|---|---|
| `Dockerfile` | **可选**：MUSA 版配方（torch_musa + MCCL） | 你侧的 MUSA 基础镜像（`FROM` 处填入自己的 registry/image tag） |
| `Dockerfile.cuda` | **可选**：CUDA 参照环境 | `pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime` |

两份镜像共用同一份钉版依赖清单（`scripts/requirements-nuplan-x86.txt`）与同一套管线脚本；
数据、地图、checkpoint 一律运行时挂载，不进镜像（nuPlan 条款 + 体积）。

MUSA 版构建要点（详见 `docs/REPRODUCE.md` 与 `docs/debug-log.md`）：
- 依赖装进 `/workspace/pylibs` 覆盖层，PYTHONPATH 顺序 = pylibs:repo:devkit，镜像自带 torch/torch_musa 不被遮蔽
- pip 24 垫底（新 pip 拒绝旧包元数据）
- 预装 pip 24 后再装清单；PL/timm/mmengine 走 `--no-deps`

```bash
# MUSA 版（填入你的基础镜像后）
docker build -f Dockerfile -t dp-nuplan-mini:musa .
docker run --gpus all -v $DATA_ROOT:/data/sets/nuplan:ro -it dp-nuplan-mini:musa bash
```
