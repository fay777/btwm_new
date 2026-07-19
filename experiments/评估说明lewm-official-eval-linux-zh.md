# LeWM 官方 Checkpoint 评测迁移说明（Linux）

本文基于 2026-07-18 已在服务器上成功跑通的流程整理，目标是：

- 在独立 `lewm` 环境中安装 LeWM 依赖
- 使用官方 `reacher` checkpoint 做本地评测
- 在新服务器上快速迁移，避免再次踩到之前的问题

## 1. 适用范围

当前这份流程只针对：

- 模型：`LeWM official checkpoint`
- 任务：`reacher`
- 评测方式：`本地 checkpoint + 本地 HDF5 数据 + CEM planner`

它用于：

- `BTWM vs LeWM` 的 cross-framework comparison

它不用于：

- LeWM 正式训练
- `BTWM vs DreamerV3` 主结果

## 2. 基本原则

- `DreamerV3 / BTWM` 环境不要和 `LeWM` 混用
- LeWM 单独使用 `conda` 环境 `lewm`
- 优先使用 `官方 checkpoint + 本地 eval`
- 不要再把 `bash code/run_lewm_dmc.sh` 当成官方 checkpoint 评测入口

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

检查版本：

```bash
python - <<'PY'
import huggingface_hub, transformers
print("huggingface_hub =", huggingface_hub.__version__)
print("transformers =", transformers.__version__)
PY
```

## 6. 推荐目录结构

```bash
/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/
  checkpoints/
  datasets/
    reacher.h5
    dmc/
      reacher_random.h5 -> /home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/datasets/reacher.h5
  hf_reacher/
    config.json
    weights.pt
  reacher/
    lewm_object.ckpt
```

说明：

- `datasets/reacher.h5` 是已有 DMC `reacher` 数据
- `datasets/dmc/reacher_random.h5` 是官方 `reacher` eval 配置期望的名字
- `reacher/lewm_object.ckpt` 是把 Hugging Face 官方模型转换后的本地 checkpoint

## 7. 下载官方 checkpoint

建议使用 Hugging Face 镜像：

```bash
cd /home/zhangpeiying01/lfy/btwm-7.7
conda activate lewm

export STABLEWM_HOME=/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache
export HF_ENDPOINT=https://hf-mirror.com

hf download quentinll/lewm-reacher --local-dir "$STABLEWM_HOME/hf_reacher"
```

检查结果：

```bash
ls -lh "$STABLEWM_HOME/hf_reacher"
```

至少应看到：

- `config.json`
- `weights.pt`

## 8. 转换官方 checkpoint

不要再手写 heredoc 转换脚本。统一使用：

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

成功时应输出：

```text
saved to /home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/reacher/lewm_object.ckpt
```

检查：

```bash
ls -lh "$STABLEWM_HOME/reacher"
```

## 9. 准备评测数据路径

当前官方 `reacher` eval 配置见：

- [reacher.yaml](E:/课题组2025-2026/具身智能/btwm-7.7/lewm_base/config/eval/reacher.yaml:1)

其中默认数据名为：

```yaml
eval:
  dataset_name: dmc/reacher_random
```

因此评测时会去找：

```bash
$STABLEWM_HOME/datasets/dmc/reacher_random.h5
```

如果你已有真实数据：

```bash
/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/datasets/reacher.h5
```

则创建软链接：

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

## 10. 正确的官方评测入口

不要直接使用：

```bash
python lewm_base/eval.py --config-name=reacher policy=...
```

原因是原始 `eval.py` 内部会调用：

```python
swm.wm.utils.load_pretrained(cfg.policy)
```

这一步可能把 `policy=...` 当成 Hugging Face repo id，导致即使本地已有 checkpoint，依然联网下载。

仓库中已新增本地评测脚本：

- [code/run_lewm_official_eval.py](E:/课题组2025-2026/具身智能/btwm-7.7/code/run_lewm_official_eval.py:1)

它会直接从绝对路径加载本地 `_object.ckpt`，绕开上面的远程解析逻辑。

## 11. 正式运行命令

后台评测：

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

前台小规模试跑：

```bash
cd /home/zhangpeiying01/lfy/btwm-7.7
conda activate lewm

export STABLEWM_HOME=/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache
export MUJOCO_GL=egl
export CUDA_VISIBLE_DEVICES=5

python code/run_lewm_official_eval.py \
  --lewm-root /home/zhangpeiying01/lfy/btwm-7.7/lewm_base \
  --config-name reacher \
  --policy-path /home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/reacher/lewm_object.ckpt \
  --overrides eval.num_eval=5
```

查看日志：

```bash
tail -f /home/zhangpeiying01/lfy/btwm-7.7/logs/lewm_eval_reacher_official.log
```

## 12. 已成功跑通的结果

2026-07-18 已在服务器上成功跑通：

- 模型：`LeWM official checkpoint`
- 任务：`reacher`
- 评测次数：`50`
- 成功率：`80.0%`

日志关键输出：

```text
1760000 valid starting points found for evaluation.
{'success_rate': 80.0, ...}
```

这说明：

- 数据加载成功
- 本地 checkpoint 加载成功
- CEM planner 正常运行
- 结果可直接用于 `BTWM vs LeWM` 的 cross-framework comparison

## 13. 常见问题与规避方法

### 13.1 不要把训练脚本当成官方评测入口

下面这条是训练，不是官方 checkpoint eval：

```bash
bash code/run_lewm_dmc.sh
```

它适合：

- 训练 smoke test
- 连通性检查
- HDF5 离线训练兼容性验证

### 13.2 不要再手敲长 heredoc

原因：

- 容易粘贴损坏
- 容易混入残缺字符
- 容易因为 HF 配置字段和本地构造函数不一致而报错

统一使用：

```bash
python code/convert_lewm_hf_ckpt.py ...
```

### 13.3 `huggingface-hub` 不要装到 `1.x`

如果报：

```text
transformers requires huggingface-hub<1.0
```

执行：

```bash
pip install "huggingface-hub>=0.34,<1.0"
```

### 13.4 缺少 `hdf5plugin` 会导致 HDF5Dataset 导入失败

如果报：

```text
ModuleNotFoundError: No module named 'hdf5plugin'
```

执行：

```bash
pip install hdf5plugin
```

### 13.5 评测数据路径错了会报 `FileNotFoundError`

如果日志在找：

```bash
.../datasets/dmc/reacher_random.h5
```

不要先乱改配置，先检查软链接和真实文件是否都存在。

### 13.6 如果又出现 Hugging Face 下载日志，说明没有命中本地模型

如果日志出现：

```text
Downloading ... from HuggingFace...
Fetching https://huggingface.co/...
```

不要继续折腾 `policy=...` 名字，直接改用：

```bash
python code/run_lewm_official_eval.py ...
```

## 14. 对比实验中的定位

LeWM 当前这条线应报告为：

- `cross-framework comparison`
- `official pretrained baseline`

不要和 `BTWM vs DreamerV3` 的同框架在线训练主结果混成同一类 baseline。

建议至少单独记录：

- `success_rate`
- `episode return`（如可导出）
- `final distance`（如可导出）
