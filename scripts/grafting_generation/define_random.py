import random
import numpy as np
import os
import secrets

# Optional imports, retained for compatibility with PyTorch, TensorFlow, and related libraries.
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
    Seed the random-number generators used by common numerical libraries.

    A shared seed makes subsequent stochastic sampling reproducible.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)  # Fix Python hash randomization.
    random.seed(seed)                         # Python standard library.
    np.random.seed(seed)                      # NumPy

    if torch is not None:
        torch.manual_seed(seed)               # CPU
        torch.cuda.manual_seed_all(seed)      # All visible GPUs, when CUDA is available.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if tf is not None:
        tf.random.set_seed(seed)

    if sklearn is not None:
        # Most scikit-learn estimators require an explicit random_state; no process-wide seed is provided.
        sklearn_set_config(transform_output='pandas')

    if transformers is not None:
        transformers.set_seed(seed)

    # LightGBM and XGBoost require a model-level seed and cannot be seeded globally.

    print(f"Global random seed: {seed}")

def init_seed(seed = None) -> int:
    """
    Initialize the global random state and return the seed that was applied.

    When ``seed`` is ``None``, a seed is drawn uniformly from ``[0, 2**31)``.
    The selected value is printed so that it can be recorded in the run log.
    """
    if seed is None:
        # Draw a seed from [0, 2**31), a range accepted by most numerical libraries.
        seed = secrets.randbelow(2**31)
        # Alternatively, secrets.randbits(32) yields an unsigned 32-bit integer.

    set_global_seed(seed)
    print(f"Global random seed:  {seed}")
    return seed
