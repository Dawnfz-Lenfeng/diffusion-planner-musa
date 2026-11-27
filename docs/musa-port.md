# MUSA (S5000) 移植说明

目标：训练与 planner 推理跑在 MTT S5000 级 MUSA GPU（torch_musa + MCCL）。
原则：**只加一层兼容模块，CUDA 路径字节级不变**；镜像自带 torch/torch_musa 不动。

## 改动清单（本仓库内）

| 文件 | 改动 |
|---|---|
| `diffusion_planner/utils/accelerator.py` | **新增**：detect_backend / device_count / set_device / synchronize / dist_backend / rank_device / assert_available，cuda/musa/cpu 三态 |
| `diffusion_planner/utils/ddp.py` | `torch.cuda.set_device` → `accelerator.set_device`；`nccl` 写死 → `accelerator.dist_backend`（musa→mccl）；SLURM 分支 device_count 走兼容层 |
| `diffusion_planner/train_epoch.py` | 2 处 `torch.cuda.synchronize()` → `accelerator.synchronize(args.device)` |
| `train_predictor.py` | 入口 `args.device = accelerator.detect_backend(args.device)`（musa 缺 torch_musa 时优雅回退 cpu）；`.to(rank if cuda else device)` → `.to(accelerator.rank_device(...))`（musa 必须用 `torch.device('musa:N')`，裸 int 会落回 cuda 语义） |
| `diffusion_planner/planner/planner.py` | device 断言扩到 musa + 可用性检查 |
| `torch_run_musa.sh` | MUSA 训练启动器（torchrun + MCCL 环境变量，沿 v4.1 已验证约定） |

torch_musa 注册 `torch.musa` 命名空间后，代码里其余的字符串设备用法
（`.to('musa')`、`torch.device('musa')`、ModelEma(device=...)）无需改动。

## 启动约定（v4.1 已验证）

```
export MCCL_IB_DISABLE=1
export PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True
python3 -m torch.distributed.run --nproc-per-node N --standalone train_predictor.py --device musa ...
```

## 验证阶梯（S5000 Pod 内）

1. `python3 -c "import torch_musa; print(torch.musa.device_count())"` ≥1
2. 单卡 CPU 语义冒烟：`--device cpu --ddp false --batch_size 8 --train_epochs 1`
3. 单卡 MUSA 冒烟：`bash torch_run_musa.sh 1 --ddp false --batch_size 8 --train_epochs 1`
4. 多卡 MUSA：`bash torch_run_musa.sh 8 --batch_size 512`（官方 500 epochs 口径）

## 风险与预案

| 风险 | 预案 |
|---|---|
| MCCL 后端名/行为差异 | `dist_backend('musa')='mccl'`；若 init 报错，先查 torch_musa 版本的文档字符串，v4.1 已验证 torchrun+MCCL 可用 |
| numpy ABI（镜像 torch_musa 编译基线 vs pylibs numpy 1.23.4） | 若 import torch 后 numpy 冲突，把 pylibs 里 numpy 删掉用镜像自带版本（devkit 对 numpy 下限宽容） |
| `nn.MultiheadAttention` 反向未实现/性能差 | torch_musa 已支持；若个别算子走 CPU 回退，batch 减档即可（模型仅 ~10M 参数） |
| checkpoint 加载 `map_location` | train_utils.resume_model 用 `map_location=device`（字符串 musa 合法） |
| pin_memory | 宿主侧固定内存，与加速卡无关，保持 True |
