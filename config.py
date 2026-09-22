"""Configuration for the final DiagScale/AMG/shared-head MSE pipeline."""
import os

def _env_path(name, default):
    return os.environ.get(name, default)

def _env_float(name, default):
    value = os.environ.get(name)
    return default if value is None else float(value)

def _env_int(name, default):
    value = os.environ.get(name)
    return default if value is None else int(value)

def _env_int_list(name, default):
    value = os.environ.get(name)
    if value is None:
        return list(default)
    value = value.strip()
    if not value:
        return list(default)
    value = value.strip('[]()')
    tokens = value.replace(';', ',').replace(' ', ',').split(',')
    dims = [int(token) for token in tokens if token.strip()]
    if len(dims) != 3:
        raise ValueError(f'{name} must contain exactly 3 integer dimensions')
    return dims

def _env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'y', 'on'}

def _env_optional_text(names, default):
    if isinstance(names, str):
        names = (names,)
    for name in names:
        value = os.environ.get(name)
        if value is None:
            continue
        value = value.strip()
        if value.lower() in {'', 'none', 'raw', 'false', '0', 'off'}:
            return None
        return value
    return default
N_ITER = _env_int('FEMGRL_N_ITER', 5)
SAVE_DIR = _env_path('FEMGRL_SAVE_DIR', '/public/home/zzx/gnn/FemGRL-3D-edgeattr/saved_models/FSS-1-T-freq')
OUTPUT_DIR = _env_path('FEMGRL_OUTPUT_DIR', '/public/home/zzx/gnn/FemGRL-3D-edgeattr/training_outputs/FSS-1-T-freq')
PRECOND_TRAIN_SCALE = _env_float('FEMGRL_PRECOND_TRAIN_SCALE', 100.0)
PRECOND_METHOD = _env_optional_text(('FEMGRL_PRECOND_METHOD', 'AMG_PRECOND_METHOD'), 'DiagScale')
DATASET_TYPE = os.environ.get('FEMGRL_DATASET_TYPE', 'FSS').upper()
DATA_ROOT_PATH = _env_path('FEMGRL_DATA_ROOT_PATH', 'data/FSS-1')
if DATASET_TYPE == 'A':
    SCA_PREFIX = 'scaA'
    DATASET_DIRS_PATTERN = 'A-TainDataset*'
elif DATASET_TYPE == 'B':
    SCA_PREFIX = 'scaB'
    DATASET_DIRS_PATTERN = 'B-TainDataset*'
elif DATASET_TYPE == 'C':
    SCA_PREFIX = 'scaC'
    DATASET_DIRS_PATTERN = 'C-TainDataset*'
elif DATASET_TYPE == 'FSS':
    SCA_PREFIX = 'FSS'
    DATASET_DIRS_PATTERN = 'FSS_case*_txt'
else:
    raise ValueError("DATASET_TYPE must be one of: 'A', 'B', 'C', 'FSS'")
DEFAULT_TARGET_GPUS = '0,1,2,3,4,5'
MASTER_PORT = '10970'
NETWORK_USE_CUSTOM_DIMS = _env_bool('FEMGRL_NETWORK_USE_CUSTOM_DIMS', True)
NETWORK_BASE_DIM = _env_int('FEMGRL_NETWORK_BASE_DIM', 64)
NETWORK_CUSTOM_DIMS = _env_int_list('FEMGRL_NETWORK_CUSTOM_DIMS', [64, 128, 256])
NETWORK_MODEL_VARIANT = os.environ.get('FEMGRL_MODEL_VARIANT', 'edge_attention').strip().lower()
NETWORK_EDGE_ATTENTION_TYPE = os.environ.get('FEMGRL_EDGE_ATTENTION_TYPE', 'mlp').lower()
NETWORK_USE_GEOMETRIC_EDGE_FEATURES = _env_bool('FEMGRL_USE_GEOMETRIC_EDGE_FEATURES', True)
NETWORK_SHARED_REAL_IMAG_HEADS = _env_bool('FEMGRL_SHARED_REAL_IMAG_HEADS', True)
NETWORK_USE_FREQUENCY_MODULATION = _env_bool('FEMGRL_USE_FREQUENCY_MODULATION', True)
FREQUENCY_FILE_PREFIX = os.environ.get('FEMGRL_FREQUENCY_FILE_PREFIX', 'Freq')
FREQUENCY_INPUT_SCALE = _env_float('FEMGRL_FREQUENCY_INPUT_SCALE', 1000000000.0)
FREQUENCY_NORM_MIN = _env_float('FEMGRL_FREQUENCY_NORM_MIN', 2.0)
FREQUENCY_NORM_MAX = _env_float('FEMGRL_FREQUENCY_NORM_MAX', 3.8)
FREQUENCY_USE_FOURIER_FEATURES = _env_bool('FEMGRL_FREQUENCY_USE_FOURIER_FEATURES', True)
FREQUENCY_EMBED_DIM = int(os.environ.get('FEMGRL_FREQUENCY_EMBED_DIM', '64'))
FREQUENCY_FILM_STRENGTH = _env_float('FEMGRL_FREQUENCY_FILM_STRENGTH', 0.1)
GRAPH_CACHE_DIR = _env_path('FEMGRL_GRAPH_CACHE_DIR', '')
GRAPH_CACHE_VERSION = os.environ.get('FEMGRL_GRAPH_CACHE_VERSION', 'v2')
GRAPH_CACHE_ENABLED = _env_bool('FEMGRL_USE_GRAPH_CACHE', True)
GRAPH_CACHE_WRITE_MISSING = _env_bool('FEMGRL_GRAPH_CACHE_WRITE_MISSING', False)
GRAPH_CACHE_STRICT = _env_bool('FEMGRL_GRAPH_CACHE_STRICT', False)
EXCLUDE_K_FILE = _env_path('FEMGRL_EXCLUDE_K_FILE', '')
EDGE_FEATURE_MATRIX = os.environ.get('FEMGRL_EDGE_FEATURE_MATRIX', 't').lower()
EDGE_ALGEBRAIC_FEATURE_DIM = _env_int('FEMGRL_EDGE_ALGEBRAIC_FEATURE_DIM', 2)
MATERIAL_WEIGHTED_LOSS = _env_bool('FEMGRL_USE_MATERIAL_WEIGHTED_LOSS', False)
AMP_ENABLED = _env_bool('FEMGRL_USE_AMP', False)
AMP_DTYPE = os.environ.get('FEMGRL_AMP_DTYPE', 'bf16').strip().lower()
if AMP_DTYPE == 'bfloat16':
    AMP_DTYPE = 'bf16'
elif AMP_DTYPE == 'float16':
    AMP_DTYPE = 'fp16'
if AMP_DTYPE not in {'bf16', 'fp16'}:
    raise ValueError('FEMGRL_AMP_DTYPE must be one of: bf16, bfloat16, fp16, float16')
AMP_USE_GRAD_SCALER = _env_bool('FEMGRL_AMP_USE_GRAD_SCALER', AMP_DTYPE == 'fp16')
LOAD_PRETRAINED_MODEL = _env_bool('FEMGRL_LOAD_PRETRAINED_MODEL', False)
PRETRAINED_MODEL_DIR = _env_optional_text('FEMGRL_PRETRAINED_MODEL_DIR', None)
CHECKPOINT_SAVE_ENABLED = _env_bool('FEMGRL_CHECKPOINT_SAVE', True)
CHECKPOINT_RESUME_ENABLED = _env_bool('FEMGRL_CHECKPOINT_RESUME', True)
CHECKPOINT_EVERY_EPOCHS = max(1, _env_int('FEMGRL_CHECKPOINT_EVERY_EPOCHS', 10))
CHECKPOINT_PATH = _env_path('FEMGRL_CHECKPOINT_PATH', '')
CHECKPOINT_RESUME_WITH_PRETRAIN = _env_bool('FEMGRL_CHECKPOINT_RESUME_WITH_PRETRAIN', False)
CHECKPOINT_VERBOSE = _env_bool('FEMGRL_CHECKPOINT_VERBOSE', False)
if NETWORK_MODEL_VARIANT != 'edge_attention':
    raise ValueError('Final training requires FEMGRL_MODEL_VARIANT=edge_attention')
if NETWORK_EDGE_ATTENTION_TYPE != 'mlp':
    raise ValueError('Final training requires FEMGRL_EDGE_ATTENTION_TYPE=mlp')
if not NETWORK_SHARED_REAL_IMAG_HEADS:
    raise ValueError('Final training requires shared real/imag heads')
if MATERIAL_WEIGHTED_LOSS:
    raise ValueError('Final training uses unweighted MSE')
if PRECOND_METHOD != 'DiagScale':
    raise ValueError('Final training requires FEMGRL_PRECOND_METHOD=DiagScale')
