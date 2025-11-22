import random
import numpy as np
import os
import secrets

# 可选：用于兼容 PyTorch、TensorFlow、其他深度学习库
try:
    import torch
except ImportError:
    torch = None

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import sklearn
    from sklearn import set_config as sklearn_set_config
except ImportError:
    sklearn = None
    sklearn_set_config = None

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import transformers
except ImportError:
    transformers = None

def set_global_seed(seed: int):
    """
    设置所有常见随机数生成库的随机种子，确保结果可复现。
    """
    os.environ['PYTHONHASHSEED'] = str(seed)  # 控制Python哈希随机性
    random.seed(seed)                         # Python标准库
    np.random.seed(seed)                      # NumPy

    if torch is not None:
        torch.manual_seed(seed)               # CPU
        torch.cuda.manual_seed_all(seed)      # 所有GPU（若可用）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if tf is not None:
        tf.random.set_seed(seed)

    if sklearn is not None:
        # sklearn 多数函数需手动传 random_state，未提供全局方法
        sklearn_set_config(transform_output='pandas')

    if transformers is not None:
        transformers.set_seed(seed)

    # LightGBM 和 XGBoost 多数模型需手动传 seed，无法全局设置

    print(f"Global random seed: {seed}")

def init_seed(seed = None) -> int:
    """
    如果 seed=None 则自动生成一个随机种子，设置全局随机状态并打印种子。
    返回最终使用的 seed（便于记录 / 日志）。
    """
    if seed is None:
        # 生成 0 .. 2**31-1 范围内的随机种子（适配大多数库）
        seed = secrets.randbelow(2**31)
        # 你也可以使用 secrets.randbits(32) 来获得 32-bit 值

    set_global_seed(seed)
    print(f"Global random seed:  {seed}")
    return seed
