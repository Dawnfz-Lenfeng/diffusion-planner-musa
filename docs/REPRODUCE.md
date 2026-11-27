# 环境搭建与全流程复现指南

目标环境：搭载 MTT S5000 MUSA GPU 的 x86_64 训练机（本基线实测硬件；其它 MUSA 卡型未验证），基础镜像含
**torch 2.7.x + torch_musa 工具链**（客户侧 MUSA 镜像均可）。纯 CPU 机器亦可完成
数据预处理与评测冒烟（planner `device=cpu`）。

## 0. 目录约定

```bash
export WORK=/data/dp-nuplan          # 数据/缓存/产出的根目录（按需修改）
export REPO=$WORK/repo               # 本仓库
export DATA_ROOT=$WORK/data/nuplan   # nuPlan 数据根
```

## 1. 数据与官方权重

```bash
bash scripts/download_data.sh "$DATA_ROOT"
# 产出：
#   $DATA_ROOT/data/nuplan/maps_zips/nuplan-maps-v1.0.zip     (971,557,640 B)
#   $DATA_ROOT/data/nuplan/maps_zips/nuplan-v1.1_mini.zip     (8,550,103,030 B)
#   $WORK/checkpoints/{args.json, model.pth}
```

下载脚本内置字节数校验与断点续传；源为 nuPlan 官方 CDN 与 HuggingFace（公开直链）。
使用数据前请确认已满足 nuPlan 官方条款（注册与合规要求见 nuscenes.org/nuplan）。

解压：

```bash
cd $DATA_ROOT && mkdir -p maps nuplan-v1.1/splits/mini
unzip -q -o maps_zips/nuplan-maps-v1.0.zip -d maps_tmp && mv maps_tmp/maps/* maps/ && rm -rf maps_tmp
unzip -q -o maps_zips/nuplan-v1.1_mini.zip -d unzip_mini
find unzip_mini -name "*.db" -exec cp -n {} nuplan-v1.1/splits/mini/ \;
# （约定：$DATA_ROOT/nuplan-v1.1/trainval 为指向 splits/mini 的软链接亦可）
```

预期：mini = 64 个 `.db` 日志；地图 4 城市（约 1.4GB）。

## 2. Python 环境

- Python **3.10**（3.9 亦可，注意依赖清单差异）
- 依赖清单：`scripts/requirements-nuplan-x86.txt`（基于 devkit 官方 requirements 的
  兼容调整，每处调整均有注释说明原因）

```bash
pip install --target=$WORK/pylibs --upgrade -r scripts/requirements-nuplan-x86.txt
# torch 栈（CUDA 机器）：
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
# MUSA 机器：使用基础镜像自带的 torch/torch_musa，勿用 pip 安装 torch
pip install --target=$WORK/pylibs --no-deps pytorch-lightning==2.0.1 timm==1.0.10 mmengine tensorboard==2.11.2
pip install --target=$WORK/pylibs --no-deps third_party/nuplan-devkit   # 见下
cp -r diffusion_planner $WORK/pylibs/
export PYTHONPATH=$WORK/pylibs:$REPO:$REPO/third_party/nuplan-devkit:$PYTHONPATH
```

### nuplan-devkit（不改动，锁定 commit）

```bash
git clone https://github.com/motional/nuplan-devkit.git third_party/nuplan-devkit
# 锁定 master @ e9241677997dd86bfc0bcd44817ab04fe631405b
```

注意：devkit 以**源码树**参与 PYTHONPATH（`pip install --target` 的副本会丢失
hydra 需要的 yaml 包数据）。

### 已知坑（详见 docs/debug-log.md）

- pip ≥25 会拒绝 omegaconf 2.0.x 等旧元数据 → 先用 pip 24.0 装依赖
  （`pip install --target=$WORK/pylibs pip==24.0` 后以 `PYTHONPATH=$WORK/pylibs python3 -m pip` 调用）
- opencv 需 headless 变体（GUI 版依赖 libGL）
- --no-deps 安装的包（mmengine/PL/timm）需手工补齐传递依赖（清单见 debug-log）
- 不要把 pylibs 的 numpy 遮蔽镜像/系统的 numpy（dist-info 也会骗过版本检查）

## 3. 数据预处理

```bash
python scripts/make_mini_log_list.py $DATA_ROOT/nuplan-v1.1/splits/mini mini_logs.json
python data_process.py \
    --data_path $DATA_ROOT/nuplan-v1.1/splits/mini \
    --map_path  $DATA_ROOT/maps \
    --save_path $WORK/cache/mini_train \
    --log_list  mini_logs.json \
    --total_scenarios 1000000 \
    --num_proc 64
# 产出：$WORK/cache/mini_train/*.npz + diffusion_planner_training.json
# 失败场景隔离记录在 failed_tokens.txt（官方串行全量约需 600h，此补丁 64 进程约 3.5h）
```

## 4. 训练（MUSA / CUDA 通用）

```bash
# MUSA 8 卡（镜像自带 torch_musa；环境约定 MCCL_IB_DISABLE=1）
bash torch_run_musa.sh 8 --batch_size 2048 --train_epochs 500
# CUDA 机器：bash torch_run.sh（上游原版启动器）
# 冒烟：bash torch_run_musa.sh 1 --ddp false --batch_size 8 --train_epochs 1 --warm_up_epoch 1
```

官方口径：8 卡 × batch 2048（每卡 256）、500 epochs、AdamW lr 5e-4。

## 5. 闭环评测（两阶段）

```bash
FILTER=mini_eval DEVICE=musa \
CKPT=<checkpoint.pth> ARGS=<训练时保存的 args.json> \
bash scripts/sim_eval.sh
python scripts/read_scores.py <metric_run_dir>
```

- 阶段 1：闭环仿真（场景日志序列化）
- 阶段 2：`run_metric.py` 计算指标 + 聚合（devkit master 为两阶段设计）
- `mini_eval.yaml` 由 `scripts/make_mini_filters.py` 生成（本交付已含生成结果，
  可自行重新生成）；官方权重评测请用 `configs/official_args.json`

CPU 评测注意：大核数机器务必设置 `OMP_NUM_THREADS=4`（默认全核线程池会使
小模型推理显著变慢）。

## 6. 可视化（可选）

`run_nuboard.ipynb`：将 notebook 中路径指向评测 run 目录（含 `.nuboard` 文件）
后于 Jupyter 中启动 bokeh 面板，回放轨迹与逐项指标。
