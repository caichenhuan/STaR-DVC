# 项目流程阅读指南

这份代码仓库现在已经整理成适合公开发布的版本。官方 release 实际只使用 `spotting.py` 和 `captioning.py` 两个模块，顺序是先 `spotting`，再 `captioning`。

## 推荐阅读顺序

1. `README.md`
2. `run_base.sh`
3. `main.py`
4. `spotting.py` / `captioning.py`
5. `dataset.py`
6. `model.py` / `model_gpt.py` / `temporal_model.py`
7. `train.py`

## 整体流程

`main.py` 现在支持显式阶段选择：

- `--stage full`：按 `spotting -> caption -> dvc` 执行完整流程
- `--stage spotting`：只跑事件定位
- `--stage caption`：只跑字幕生成训练
- `--stage dvc`：只跑最终密集视频字幕生成

其中：

- `spotting` 产出事件定位结果
- `caption` 负责字幕模型训练
- `dvc` 使用 `caption` 权重和 `spotting` 输出生成最终 dense captions

## 各阶段关键文件

### 1. 事件定位

- 入口：`spotting.py`
- 模型：`model.py` 中的 `Video2Spot`
- 作用：在整场比赛时间线上做事件时间定位

### 2. 字幕生成训练

- 入口：`captioning.py` 中的 `main(args)`
- 模型：`model_gpt.py` 中的 `Video2CaptionGPT`
- 解码器：`gpt.py`
- 作用：在 ground-truth 事件片段上训练 caption generator

### 3. DVC 推理与评估

- 入口：`captioning.py` 中的 `dvc(args)`
- 输入：`spotting` 阶段的预测窗口
- 输出：dense video captions 以及 tight / medium / loose 评测结果

## 数据流怎么走

`dataset.py` 里最重要的是三个数据集：

- `SoccerNetDataset`
- `SoccerNetClipsTesting`
- `PredictionCaptions`

它们分别覆盖：

- 训练阶段的样本构造
- spotting 测试时的整场滑窗
- DVC 阶段基于 spotting 结果的字幕推理

## 模型核心

`model.py` 中的 `VideoEncoder` 是共享主干，核心池化器在 `temporal_model.py`：

- 局部视频片段特征
- 整个半场的全局特征
- 2D 球员位置特征

这三类信息最终在 `PerceiverResamplerGlobalPosition` 中融合。

## 训练逻辑

`train.py` 是通用训练与评估入口，建议重点看：

- `trainer()`
- `train()`
- `validate_captioning()`
- `validate_spotting()`
- `test_spotting()`
- `test_captioning()`

## 当前整理版特别说明

1. 公开版默认关闭了 `efficient_memory` 实验路径。
原因：当前实现仍偏实验性质，不适合作为默认主线。

2. `run_base.sh` 和 `run_debug.sh` 已改成环境变量驱动。
原因：原始脚本里写的是私有机器路径，公开仓库不能保留。

3. `dataset.py` 末尾的本地测试片段已移除。
原因：这些内容依赖私有目录，保留会误导使用者。

4. `classifying.py` 保留在仓库里，但不属于官方 release 主流程。
原因：最终公开版本只使用 `spotting.py` 和 `captioning.py`。

## 快速定位代码

- 看流程：`main.py`
- 看数据：`dataset.py`
- 看模型：`model.py` + `model_gpt.py` + `temporal_model.py`
- 看训练：`train.py`
- 看运行参数：`run_base.sh`
