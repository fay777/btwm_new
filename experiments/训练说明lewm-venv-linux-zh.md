# LeWM `lewm` 环境安装与排错说明（Linux）

这份说明基于 2026-07-17 的实际服务器排错过程整理，目标是让 LeWM 在独立
`lewm` 环境里稳定读取 `reacher.h5` 并启动训练。

## 1. 环境建议

- 继续保留现有 `DreamerV3 / BTWM` 环境，不要混装
- LeWM 单独使用 `conda` 环境 `lewm`
- 推荐 Python `3.10`

创建环境：

```bash
conda create -n lewm python=3.10
conda activate lewm
python --version
```

## 2. 安装依赖

仓库内的依赖文件已经包含这次排错需要的关键版本：

- `stable-worldmodel[train,env]==0.1.0`
- `datasets<3`
- `pyarrow==20.0.0`
- `hdf5plugin`

安装：

```bash
pip install -U pip setuptools wheel
pip install -r lewm_base/requirements-linux-btwm.txt
```

等价的关键包如下：

```bash
pip install "stable-worldmodel[train,env]==0.1.0"
pip install stable-pretraining
pip install "datasets<3" "pyarrow==20.0.0" hdf5plugin
pip install einops hydra-core lightning scikit-learn
```

## 3. 数据目录约定

LeWM 当前 DMC 入口默认支持：

- `reacher`
- `reacher_easy`

数据文件应当是：

- `reacher.h5`

推荐目录结构：

```bash
/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/
  datasets/
    reacher.h5
```

也就是说，真正传给训练的数据目录应该是 `datasets/` 这一层。

准备方式：

```bash
mkdir -p /home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/datasets
cp /path/to/reacher.h5 \
  /home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/datasets/reacher.h5
```

## 4. 启动前检查

### 4.1 依赖检查

```bash
python - <<'PY'
import stable_worldmodel as swm
import stable_worldmodel.data as data
print("stable_worldmodel:", swm.__file__)
print("has HDF5Dataset:", hasattr(data, "HDF5Dataset"))
PY
```

如果 `has HDF5Dataset` 是 `False`，继续检查底层实现是否存在：

```bash
grep -R "HDF5Dataset" ~/.conda/envs/lewm/lib/python3.10/site-packages/stable_worldmodel -n
```

### 4.2 HDF5 读取依赖检查

```bash
python - <<'PY'
try:
    from stable_worldmodel.data.formats.hdf5 import HDF5Dataset
    print("HDF5Dataset import ok")
except Exception as e:
    import traceback
    traceback.print_exc()
PY
```

如果这里报 `ModuleNotFoundError: No module named 'hdf5plugin'`，执行：

```bash
pip install hdf5plugin
```

### 4.3 数据文件检查

```bash
ls -lh /home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/datasets/reacher.h5
```

## 5. 运行方式

先进入仓库根目录，不要在别的目录直接执行相对路径：

```bash
cd /home/zhangpeiying01/lfy/btwm-7.7
chmod +x code/run_lewm_experiment.sh code/run_lewm_dmc.sh
```

### 5.1 Dry-run

```bash
export LEWM_PYTHON=python
export LEWM_STABLEWM_HOME=/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache
export LEWM_DATASET_DIR=/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/datasets
export GPU=5
export TASK=reacher
export SEED=0
export DRY_RUN=1

bash code/run_lewm_dmc.sh
```

### 5.2 首次试跑

```bash
export LEWM_PYTHON=python
export LEWM_STABLEWM_HOME=/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache
export LEWM_DATASET_DIR=/home/zhangpeiying01/lfy/btwm-7.7/btwm_data/lewm_cache/datasets
export GPU=5
export TASK=reacher
export SEED=0
export LEWM_BATCH_SIZE=32
export LEWM_MAX_EPOCHS=5
unset DRY_RUN
unset LEWM_DATASET_NAME
unset LEWM_DATASET_FORMAT

bash code/run_lewm_dmc.sh
```

### 5.3 后台运行

```bash
nohup bash code/run_lewm_dmc.sh > logs/lewm_dmc_reacher_seed0_nohup.log 2>&1 &
echo $!
tail -f logs/lewm_dmc_reacher_seed0_nohup.log
```

## 6. 这次实际修过的问题

为了让现有 `lewm_base` 兼容服务器上的 `stable-worldmodel==0.1.0`，仓库内已经做了这些修正：

1. `code/run_lewm_experiment.sh`
   - `TASK=reacher` 映射到 `data=dmc`
   - 默认传本地 HDF5 绝对路径：
     `"$LEWM_DATASET_DIR/reacher.h5"`

2. `lewm_base/config/train/data/dmc.yaml`
   - 默认数据集名改为 `reacher`

3. `lewm_base/train.py`
   - 增加 `load_dataset_compat()`
   - 如果传入的是 `.h5` 文件路径，直接走
     `stable_worldmodel.data.formats.hdf5.HDF5Dataset`
   - 传路径时使用 `path=...`，避免错误拼成
     `datasets/datasets/reacher.h5.h5`

## 7. 公平性提醒

LeWM 当前这条线仍然是：

- 离线 world-model + planner

而 BTWM / DreamerV3 是：

- 在线 RL

因此 LeWM 只能作为：

- `cross-framework comparison`

不要和 `BTWM vs DreamerV3 on walker_walk` 的主结果混成同一类 baseline。
