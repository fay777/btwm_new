# LeWM 官方 Checkpoint 评估迁移说明（Linux，V2）

本文基于 2026-07-18 在服务器上成功跑通的流程整理，目标是：

- 在独立 `lewm` 环境中安装 LeWM 依赖
- 使用官方 `reacher` checkpoint 做本地评估
- 在新服务器上迁移时，尽量避免重复踩坑

## 1. 适用范围

当前流程只针对：

- 模型：`LeWM official checkpoint`
- 任务：`reacher`
- 评估方式：`本地 checkpoint + 本地 HDF5 数据 + CEM planner`

它用于：

- `BTWM vs LeWM` 的 cross-framework comparison

它不用于：

- LeWM 正式训练
- `BTWM vs DreamerV3` 主结果

## 2. 基本原则

- `DreamerV3 / BTWM` 环境不要和 `LeWM` 混用
- LeWM 单独使用 `conda` 环境 `lewm`
- 优先使用 `官方 checkpoint + 本地 eval`
- 不要再把 `bash code/run_lewm_dmc.sh` 当成官方 checkpoint 评估入口

## 3. 创建环境

```bash
conda create -n lewm python=3.10
conda activate lewm
python --version
```

## 4. 安装依赖

推荐直接使用仓库里的依赖文件：

```bash
cd /home/zhangpeiying01/lfy/btwm-7.7
pip install -U pip setuptools wheel
pip install -r lewm_base/requirements-linux-btwm.txt
```

关键版本包括：

- `stable-worldmodel[train,env]==0.1.0`
- `stable-pretraining`
- `datasets<3`
- `pyarrow==20.0.0`
- `hdf5plugin`
- `einops`
- `hydra-core`
- `lightning`
- `scikit-learn`

如果需要手动安装：

```bash
pip install "stable-worldmodel[train,env]==0.1.0"
pip install stable-pretraining
pip install "datasets<3" "pyarrow==20.0.0" hdf5plugin
pip install einops hydra-core lightning scikit-learn
```

## 5. Hugging Face 依赖兼容性

如果 `transformers` 报 `huggingface-hub` 版本不兼容，不要忽略。

推荐固定：

```bash
pip install "huggingface-hub>=0.34,<1.0"
```

## 6. 推荐目录结构

```bash
/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/
  checkpoints/
  datasets/
    reacher.h5
    dmc/
      reacher_random.h5
  hf_reacher/
    config.json
    weights.pt
  reacher/
    lewm_object.ckpt
```

## 7. 下载官方 checkpoint

```bash
cd /home/zhangpeiying01/lfy/btwm-7.7
conda activate lewm

export STABLEWM_HOME=/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache
export HF_ENDPOINT=https://hf-mirror.com

hf download quentinll/lewm-reacher --local-dir "$STABLEWM_HOME/hf_reacher"
```

## 8. 转换官方 checkpoint

使用：

- [code/convert_lewm_hf_ckpt.py](E:/课题组2025-2026/具身智能/btwm-7.7/code/convert_lewm_hf_ckpt.py:1)

执行：

```bash
cd /home/zhangpeiying01/lfy/btwm-7.7
conda activate lewm

export STABLEWM_HOME=/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache

python code/convert_lewm_hf_ckpt.py \
  --src "$STABLEWM_HOME/hf_reacher" \
  --out "$STABLEWM_HOME/reacher/lewm_object.ckpt" \
  --lewm-root /home/zhangpeiying01/lfy/btwm-7.7/lewm_base
```

## 9. 准备评估数据路径

当前官方 `reacher` eval 配置默认数据名为：

```yaml
eval:
  dataset_name: dmc/reacher_random
```

因此默认会去找：

```bash
$STABLEWM_HOME/datasets/dmc/reacher_random.h5
```

这里提供两种方式。

### 方式 A：使用软链接

适合：

- 你已有 `datasets/reacher.h5`
- 不想改命令和配置

执行：

```bash
cd /home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache
mkdir -p datasets/dmc
ln -sf \
  /home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/datasets/reacher.h5 \
  /home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/datasets/dmc/reacher_random.h5
```

检查：

```bash
ls -lh /home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/datasets/dmc
readlink -f /home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/datasets/dmc/reacher_random.h5
```

优点：

- 不改配置
- 与官方 `reacher.yaml` 完全一致

### 方式 B：不使用软链接，直接覆盖数据名

适合：

- 不想维护软链接
- 想在命令里直接指定真实数据名

前提是真实文件在：

```bash
$STABLEWM_HOME/datasets/reacher.h5
```

运行时加：

```bash
--overrides eval.dataset_name=reacher
```

优点：

- 不需要 `datasets/dmc/reacher_random.h5`
- 数据路径更直观

注意：

- 这条方式必须显式写 `--overrides eval.dataset_name=reacher`

## 10. 正确的官方评估入口

不要直接使用：

```bash
python lewm_base/eval.py --config-name=reacher policy=...
```

因为原始 `eval.py` 会调用：

```python
swm.wm.utils.load_pretrained(cfg.policy)
```

这可能把 `policy=...` 当成 Hugging Face repo id。

统一改用：

- [code/run_lewm_official_eval.py](E:/课题组2025-2026/具身智能/btwm-7.7/code/run_lewm_official_eval.py:1)

它会直接从绝对路径加载本地 `_object.ckpt`。

## 11. 正式运行命令

### 11.1 用软链接的版本

```bash
cd /home/zhangpeiying01/lfy/btwm-7.7
conda activate lewm

export STABLEWM_HOME=/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=5

nohup python code/run_lewm_official_eval.py \
  --lewm-root /home/zhangpeiying01/lfy/btwm-7.7/lewm_base \
  --config-name reacher \
  --policy-path /home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/reacher/lewm_object.ckpt \
  > /home/zhangpeiying01/lfy/btwm-7.7/logs/lewm_eval_reacher_official.log 2>&1 &
echo $!
```

### 11.2 不用软链接的版本

```bash
cd /home/zhangpeiying01/lfy/btwm-7.7
conda activate lewm

export STABLEWM_HOME=/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=5

nohup python code/run_lewm_official_eval.py \
  --lewm-root /home/zhangpeiying01/lfy/btwm-7.7/lewm_base \
  --config-name reacher \
  --policy-path /home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/reacher/lewm_object.ckpt \
  --overrides eval.dataset_name=reacher \
  > /home/zhangpeiying01/lfy/btwm-7.7/logs/lewm_eval_reacher_official.log 2>&1 &
echo $!
```

## 12. 已成功跑通的结果

2026-07-18 已在服务器上成功跑通：

- 模型：`LeWM official checkpoint`
- 任务：`reacher`
- 评估次数：`50`
- 成功率：`80.0%`

日志关键输出：

```text
1760000 valid starting points found for evaluation.
{'success_rate': 80.0, ...}
```

## 13. 常见问题

- 如果又出现 Hugging Face 下载日志，说明没有命中本地模型，应使用 `code/run_lewm_official_eval.py`
- 如果报 `FileNotFoundError`，先确认你走的是软链接版还是非软链接版
- 如果报 `hdf5plugin` 缺失，执行 `pip install hdf5plugin`
- 如果报 `huggingface-hub` 版本冲突，执行 `pip install "huggingface-hub>=0.34,<1.0"`
