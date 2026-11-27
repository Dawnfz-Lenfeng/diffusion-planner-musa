# 环境与适配调试记录（dp-nuplan mini × Diffusion Planner × MUSA）

时间：2026-11-21 ｜ 环境：MT GPU <training-pod> Pod（S5000×8, 255C, training-suite v2.1.7-rc4_dsv4_20261030, py3.10.12, torch 2.7.1.post1 + torch_musa）

## 依赖安装（pylibs 覆盖层，不动镜像）

| 问题 | 根因 | 解法 |
|---|---|---|
| pip26 `omegaconf==2.0.6/2.1.0` No matching distribution | pip26 拒绝旧包元数据（版本列表里明明有） | **pip24 覆盖层**：`pip install --target=pylibs pip==24.0`，之后 `PYTHONPATH=pylibs python3 -m pip` 干活 |
| hydra-core==1.1.0rc1 元数据同样炸 pip26 | rc 版本串 | 换 **hydra-core==1.1.0 + omegaconf==2.1.1**（hydra 1.1.0 的真实约束 ==2.1.*，2.0.6 会被拒） |
| opencv-python 4.5.1.48 源码编译失败 | 无 cp310 wheel；GUI 版还需 libGL（镜像没有） | **opencv-python-headless==4.5.5.64** |
| mmengine/timm/PL --no-deps 后 ModuleNotFoundError | 裸 --no-deps 不带传递依赖 | 补 yapf/addict/regex/termcolor + lightning-utilities/torchmetrics |
| torchmetrics 拖进 pypi torch 遮蔽镜像 torch_musa | 补包时忘 --no-deps | 删 pylibs/{torch,torchgen,functorch,numpy…}；**numpy 用镜像的 1.26.4**（pylibs 的 1.23.4 遮蔽会炸 accelerate 的 np._core 检查） |
| numpy dist-info 残留骗版本检查 | rm 目录没删 dist-info | 清 pylibs 里所有孤儿 *.dist-info |
| devkit pip --target 副本缺 yaml 包数据 | hydra pkg:// 找不到 splitter 等 | **PYTHONPATH 顺序 repo:devkit源码树:pylibs**，nuplan 永远从源码树解析 |

## 网络

- Pod 公网白名单实测：CloudFront ✅ / pypi tuna ✅；pypi.org、aliyun、files.pythonhosted、download.pytorch.org 全 ✗
- CloudFront 出口间歇性全 000（持续 2.5h+）→ 依赖 `wget -c -T 30 -t 3` 重试 + 大文件兜底走 **stdin 流式传输**（laptop→ops-gateway ssh→`kubectl exec -i cat > file`，~2-4MB/s，9.5GB 全量校验通过）

## 训练侧（MUSA）

| 问题 | 解法 |
|---|---|
| 上游 5 处 torch.cuda 硬编码 | accelerator.py 兼容层（cuda/musa/cpu），DDP 后端 nccl↔mccl |
| `--device musa` 需 `torch.device('musa:N')`（裸 int 落回 cuda 语义） | rank_device() 统一处理 |
| **8 卡 torchrun 卡死在 init_process_group** | ddp.py 把 torchrun --standalone 已设的 MASTER_ADDR/PORT 覆盖成固定 22323，与残留进程的 store 冲突；**尊重 torchrun 提供的 rendezvous，仅在缺失时兜底**（faulthandler SIGUSR1 抓栈定位） |
| 训练冒烟 `--train_epochs 1` 触发 warm_up 断言 | 冒烟参数 `--warm_up_epoch 1` |
| 官方 args.json 早于 guidance 功能，缺 guidance_fn 键 | train_predictor 补 `--guidance_fn` 默认 None；args.json 打补丁 |

## 评测侧（nuplan-devkit master e924167）

| 问题 | 解法 |
|---|---|
| 所有场景 "Simulation failed with error:"（消息为空） | 完整 traceback 在 tee 日志里：normalizer torch.ne 收到 numpy —— **fork 并行版 data_processor 重写 observation_adapter 时丢了 `convert_to_model_inputs`**（预处理路径不用该函数，训练无感知） |
| 仿真成功但 metrics 目录恒空 | devkit master 是两阶段：`run_simulation.py` 只写 simulation_log，指标要单独跑 **run_metric.py**；且 `simulation_log_main_path` 默认 null 导致仿真日志不落盘 |
| simulation_log 目录有树无文件 | SimulationLogCallback 的 futures 进程退出即丢 → **Pod 本地补丁强制同步写盘**（scripts/patch_sync_savelog.py） |
| 指标聚合 "No metric files found" | 聚合按路径匹配 challenge 名，experiment_uid 必须含 challenge |
| `worker.threads_per_node` 不能覆盖 | sequential worker 无该字段 |
| 官方 val14/test14 filter 与 mini 不相交 | 自建 mini_one.yaml / mini_eval.yaml（scripts/make_mini_filters.py，token 扫自 mini db） |

## 纪律（教训固化）

1. 动 PyPI 依赖：能 --no-deps 就 --no-deps；不能时确认不会拖 torch/numpy
2. 删包同时删 dist-info
3. pylibs 永远放 PYTHONPATH **最后**（repo:devkit:pylibs），镜像包优先
4. 多进程卡死 → `PYTHONFAULTHANDLER=1` + faulthandler.register(SIGUSR1) 抓栈，别猜
5. executor "Simulation failed with error:" 后面跟着空异常 ≠ 没有错误——完整 traceback 在 tee 日志里
6. kubectl exec 里 nohup 启动长任务后，用日志标记位轮询确认存活，`ps` 截断会误判
