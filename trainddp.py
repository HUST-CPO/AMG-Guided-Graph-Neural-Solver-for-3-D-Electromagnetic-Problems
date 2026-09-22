from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import timedelta
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import os
import gc
import json
import time
import random
from collections import OrderedDict
from contextlib import nullcontext
from sklearn.model_selection import train_test_split
from training.training_subset import load_training_subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from training.build_graph import build_or_load_graph_data, scan_all_data, root_data_path, load_file_data, is_main_process
from training.build_graph import PRECOND_METHOD, sca_prefix
from training.model_amg_with_edge_attention import BuildGCNSharedHeadsList
from training.data_io import unpack_sample_info
import sys
REDUCE_LR_PATIENCE = 40
REDUCE_LR_FACTOR = 0.5
REDUCE_LR_MIN_LR = 1e-06
EARLY_STOPPING_ENABLED = os.environ.get('FEMGRL_DISABLE_EARLY_STOPPING', '0').strip().lower() not in {'1', 'true', 'yes', 'y', 'on'}
EARLY_STOPPING_PATIENCE = int(os.environ.get('FEMGRL_EARLY_STOPPING_PATIENCE', '100'))
EARLY_STOPPING_MIN_DELTA = float(os.environ.get('FEMGRL_EARLY_STOPPING_MIN_DELTA', '1e-5'))
EARLY_STOPPING_START_EPOCH = int(os.environ.get('FEMGRL_EARLY_STOPPING_START_EPOCH', '50'))
MATRIX_CACHE = OrderedDict()
MATRIX_CACHE_BYTES = 0
MATRIX_CACHE_HITS = 0
MATRIX_CACHE_MISSES = 0
MATRIX_CACHE_LOADS = 0
MATRIX_DATA_MAPPING = {}
MATRIX_N_TOTAL = 0
GPU_MATRIX_CACHE = OrderedDict()
GPU_MATRIX_CACHE_BYTES = 0
GPU_MATRIX_CACHE_HITS = 0
GPU_MATRIX_CACHE_MISSES = 0
VALIDATE_EVERY = max(1, int(os.environ.get('FEMGRL_VALIDATE_EVERY', '10')))
CPU_MATRIX_CACHE_LIMIT_MB = max(0, int(os.environ.get('FEMGRL_CPU_MATRIX_CACHE_LIMIT_MB', '12000')))
CPU_MATRIX_CACHE_LIMIT_BYTES = CPU_MATRIX_CACHE_LIMIT_MB * 1024 * 1024
GPU_MATRIX_CACHE_LIMIT_MB = max(0, int(os.environ.get('FEMGRL_GPU_MATRIX_CACHE_LIMIT_MB', '0')))
GPU_MATRIX_CACHE_LIMIT_BYTES = GPU_MATRIX_CACHE_LIMIT_MB * 1024 * 1024
TOTAL_BATCH_SIZE = max(1, int(os.environ.get('FEMGRL_TOTAL_BATCH_SIZE', '1')))
EPOCH_ADAM = max(1, int(os.environ.get('FEMGRL_EPOCH_ADAM', '4000')))
TOTAL_EPOCHS = max(1, int(os.environ.get('FEMGRL_TOTAL_EPOCHS', str(EPOCH_ADAM))))
EPOCH_PRINT = 50
RANDOM_SEED = int(os.environ.get('FEMGRL_RANDOM_SEED', '42'))
from training.config import N_ITER, SAVE_DIR, OUTPUT_DIR, MASTER_PORT, LOAD_PRETRAINED_MODEL, PRETRAINED_MODEL_DIR, NETWORK_SHARED_REAL_IMAG_HEADS, GRAPH_CACHE_ENABLED, GRAPH_CACHE_WRITE_MISSING, GRAPH_CACHE_STRICT, EXCLUDE_K_FILE, PRECOND_TRAIN_SCALE, NETWORK_USE_FREQUENCY_MODULATION, FREQUENCY_NORM_MIN, FREQUENCY_NORM_MAX, FREQUENCY_EMBED_DIM, FREQUENCY_FILM_STRENGTH, AMP_ENABLED, AMP_DTYPE, AMP_USE_GRAD_SCALER, CHECKPOINT_SAVE_ENABLED, CHECKPOINT_RESUME_ENABLED, CHECKPOINT_EVERY_EPOCHS, CHECKPOINT_PATH, CHECKPOINT_RESUME_WITH_PRETRAIN, CHECKPOINT_VERBOSE

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_model_parameter_dtype(model, default=torch.float32):
    for param in model.parameters():
        return param.dtype
    return default

def get_amp_torch_dtype():
    if AMP_DTYPE == 'bf16':
        return torch.bfloat16
    if AMP_DTYPE == 'fp16':
        return torch.float16
    raise ValueError(f'Unsupported AMP_DTYPE: {AMP_DTYPE}')
AMP_TORCH_DTYPE = get_amp_torch_dtype()

def amp_autocast_context(device):
    if AMP_ENABLED and getattr(device, 'type', None) == 'cuda':
        return torch.autocast(device_type='cuda', dtype=AMP_TORCH_DTYPE)
    return nullcontext()

def create_amp_grad_scaler(device):
    scaler_enabled = AMP_ENABLED and AMP_USE_GRAD_SCALER and (AMP_DTYPE == 'fp16') and (getattr(device, 'type', None) == 'cuda')
    try:
        return torch.amp.GradScaler('cuda', enabled=scaler_enabled)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=scaler_enabled)

def load_excluded_k_indices(path):
    if not path:
        return set()
    if not os.path.exists(path):
        raise FileNotFoundError(f'Exclude k file not found: {path}')

    def _add_value(values, value):
        if value is None or isinstance(value, bool):
            return
        if isinstance(value, int):
            values.add(int(value))
            return
        if isinstance(value, float) and value.is_integer():
            values.add(int(value))
            return
        if isinstance(value, str):
            token = value.strip()
            if token:
                values.add(int(token))
    excluded = set()
    if path.lower().endswith('.json'):
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            candidates = []
            for key in ('bad_k_indices', 'bad_indices', 'excluded_k_indices', 'excluded', 'failed_k_indices', 'failed'):
                if key in payload:
                    candidates = payload[key]
                    break
            if not candidates and 'results' in payload:
                candidates = [item.get('k_idx') for item in payload['results'] if isinstance(item, dict) and item.get('status') not in {'ok', 'safe'}]
        else:
            candidates = payload
        for item in candidates:
            if isinstance(item, dict):
                _add_value(excluded, item.get('k_idx'))
            else:
                _add_value(excluded, item)
        return excluded
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('#', 1)[0].replace(',', ' ')
            for token in line.split():
                _add_value(excluded, token)
    return excluded

def filter_data_mapping_by_k(data_mapping, valid_k_list):
    valid_k_set = set((int(k) for k in valid_k_list))
    return {int(k): value for (k, value) in data_mapping.items() if int(k) in valid_k_set}

def get_runtime_exclude_k_file():
    return os.environ.get('FEMGRL_EXCLUDE_K_FILE', EXCLUDE_K_FILE)
LR = float(os.environ.get('FEMGRL_LR', '0.002'))
GRAD_CLIP_NORM = max(0.0, float(os.environ.get('FEMGRL_GRAD_CLIP_NORM', '1.0')))
PIN_MEMORY = False

class Logger(object):

    def __init__(self, filename='Default.log'):
        self.terminal = sys.stdout
        self.log = open(filename, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

class KIndexGraphDataset(Dataset):

    def __init__(self, k_indices, use_graph_cache=GRAPH_CACHE_ENABLED, write_missing=GRAPH_CACHE_WRITE_MISSING, strict_cache=GRAPH_CACHE_STRICT):
        self.k_indices = [int(k) for k in k_indices]
        self.use_graph_cache = bool(use_graph_cache)
        self.write_missing = bool(write_missing)
        self.strict_cache = bool(strict_cache)

    def __len__(self):
        return len(self.k_indices)

    def __getitem__(self, index):
        return build_or_load_graph_data(self.k_indices[index], use_cache=self.use_graph_cache, write_missing=self.write_missing, strict_cache=self.strict_cache)

def maybe_pin_memory(tensor):
    return tensor.pin_memory() if PIN_MEMORY else tensor

def _tensor_nbytes(tensor):
    if tensor is None:
        return 0
    return tensor.element_size() * tensor.numel()

def _matrix_entry_nbytes(A_gpu, b_gpu):
    A_gpu = A_gpu.coalesce() if getattr(A_gpu, 'is_sparse', False) else A_gpu
    if getattr(A_gpu, 'is_sparse', False):
        a_bytes = _tensor_nbytes(A_gpu._indices()) + _tensor_nbytes(A_gpu._values())
    else:
        a_bytes = _tensor_nbytes(A_gpu)
    return a_bytes + _tensor_nbytes(b_gpu)

def _device_cache_key(k_idx, device, dtype, use_original):
    device_index = device.index
    if device.type == 'cuda' and device_index is None:
        device_index = torch.cuda.current_device()
    dtype_key = str(dtype) if dtype is not None else 'default'
    return (device.type, device_index, int(k_idx), bool(use_original), dtype_key)

def _gpu_cache_get(key):
    global GPU_MATRIX_CACHE_HITS, GPU_MATRIX_CACHE_MISSES
    if GPU_MATRIX_CACHE_LIMIT_BYTES <= 0:
        GPU_MATRIX_CACHE_MISSES += 1
        return None
    entry = GPU_MATRIX_CACHE.get(key)
    if entry is None:
        GPU_MATRIX_CACHE_MISSES += 1
        return None
    GPU_MATRIX_CACHE.move_to_end(key)
    GPU_MATRIX_CACHE_HITS += 1
    return (entry['A'], entry['b'])

def _gpu_cache_put(key, A_gpu, b_gpu):
    global GPU_MATRIX_CACHE_BYTES
    if GPU_MATRIX_CACHE_LIMIT_BYTES <= 0:
        return
    entry_bytes = _matrix_entry_nbytes(A_gpu, b_gpu)
    if entry_bytes > GPU_MATRIX_CACHE_LIMIT_BYTES:
        return
    old_entry = GPU_MATRIX_CACHE.pop(key, None)
    if old_entry is not None:
        GPU_MATRIX_CACHE_BYTES -= old_entry['bytes']
    while GPU_MATRIX_CACHE and GPU_MATRIX_CACHE_BYTES + entry_bytes > GPU_MATRIX_CACHE_LIMIT_BYTES:
        (_, evicted) = GPU_MATRIX_CACHE.popitem(last=False)
        GPU_MATRIX_CACHE_BYTES -= evicted['bytes']
    GPU_MATRIX_CACHE[key] = {'A': A_gpu, 'b': b_gpu, 'bytes': entry_bytes}
    GPU_MATRIX_CACHE_BYTES += entry_bytes

def clear_gpu_matrix_cache():
    global GPU_MATRIX_CACHE_BYTES
    GPU_MATRIX_CACHE.clear()
    GPU_MATRIX_CACHE_BYTES = 0
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _cpu_matrix_entry_nbytes(entry):
    total = 0
    seen = set()
    for value in entry.values():
        if isinstance(value, torch.Tensor):
            value_id = id(value)
            if value_id not in seen:
                total += _tensor_nbytes(value)
                seen.add(value_id)
    return total

def _cpu_cache_get(k_idx):
    global MATRIX_CACHE_HITS, MATRIX_CACHE_MISSES
    k_idx = int(k_idx)
    entry = MATRIX_CACHE.get(k_idx)
    if entry is None:
        MATRIX_CACHE_MISSES += 1
        return None
    MATRIX_CACHE.move_to_end(k_idx)
    MATRIX_CACHE_HITS += 1
    return entry

def _cpu_cache_put(k_idx, entry, allow_evict=True):
    global MATRIX_CACHE_BYTES
    if CPU_MATRIX_CACHE_LIMIT_BYTES <= 0:
        return False
    k_idx = int(k_idx)
    entry_bytes = _cpu_matrix_entry_nbytes(entry)
    if entry_bytes > CPU_MATRIX_CACHE_LIMIT_BYTES:
        return False
    old_entry = MATRIX_CACHE.get(k_idx)
    old_bytes = old_entry.get('_cache_bytes', _cpu_matrix_entry_nbytes(old_entry)) if old_entry is not None else 0
    projected_bytes = MATRIX_CACHE_BYTES - old_bytes + entry_bytes
    if not allow_evict and projected_bytes > CPU_MATRIX_CACHE_LIMIT_BYTES:
        return False
    if old_entry is not None:
        MATRIX_CACHE.pop(k_idx, None)
        MATRIX_CACHE_BYTES -= old_bytes
    while MATRIX_CACHE and MATRIX_CACHE_BYTES + entry_bytes > CPU_MATRIX_CACHE_LIMIT_BYTES:
        if not allow_evict:
            return False
        (_, evicted) = MATRIX_CACHE.popitem(last=False)
        MATRIX_CACHE_BYTES -= evicted.get('_cache_bytes', _cpu_matrix_entry_nbytes(evicted))
    entry['_cache_bytes'] = entry_bytes
    MATRIX_CACHE[k_idx] = entry
    MATRIX_CACHE_BYTES += entry_bytes
    return True

def _load_matrix_entry_from_disk(k_idx, data_mapping=None):
    global MATRIX_CACHE_LOADS
    data_mapping = MATRIX_DATA_MAPPING if data_mapping is None else data_mapping
    k_idx = int(k_idx)
    if k_idx not in data_mapping:
        raise RuntimeError(f'Matrix sample k={k_idx} is not in data_mapping')
    (folder_path, name_prefix, folder_num, data_id) = unpack_sample_info(data_mapping[k_idx], sca_prefix)
    Aij = load_file_data(folder_path, 'Aij', folder_num, data_id, name_prefix)
    Av = load_file_data(folder_path, 'Av', folder_num, data_id, name_prefix)
    b_data = load_file_data(folder_path, 'b', folder_num, data_id, name_prefix)
    rows = Aij[:, 0].astype(int) - 1
    cols = Aij[:, 1].astype(int) - 1
    values = Av[:, 0] + 1j * Av[:, 1]
    n_nodes = len(b_data)
    b_val = b_data[:, 0] + 1j * b_data[:, 1]
    shape = (n_nodes, n_nodes)
    i_orig = maybe_pin_memory(torch.from_numpy(np.vstack((rows, cols))).long())
    v_orig = maybe_pin_memory(torch.from_numpy(values.astype(np.complex64)))
    b_orig = maybe_pin_memory(torch.from_numpy(b_val.astype(np.complex64)))
    precond_file = os.path.join(folder_path, f'precond_{PRECOND_METHOD}_{name_prefix}{folder_num}_{data_id}.pt') if PRECOND_METHOD else None
    if precond_file and os.path.exists(precond_file):
        try:
            pt_data = torch.load(precond_file, map_location='cpu', weights_only=True)
        except TypeError:
            pt_data = torch.load(precond_file, map_location='cpu')
        A_new = pt_data['A_new']
        if not getattr(A_new, 'is_sparse', False):
            A_new = A_new.to_sparse_coo()
        A_new = A_new.coalesce()
        b_new = pt_data['b_new']
        shape = tuple(A_new.shape)
        i_new = maybe_pin_memory(A_new._indices().long())
        v_new = maybe_pin_memory(A_new._values().to(torch.complex64))
        b_new_vec = maybe_pin_memory((b_new * PRECOND_TRAIN_SCALE).to(torch.complex64))
    else:
        (i_new, v_new, b_new_vec) = (i_orig, v_orig, b_orig)
    MATRIX_CACHE_LOADS += 1
    return {'indices': i_new, 'values': v_new, 'shape': shape, 'b': b_new_vec, 'orig_indices': i_orig, 'orig_values': v_orig, 'orig_b': b_orig}

def get_Ab(k_idx, device, dtype=None, use_original=False):
    cache_data = _cpu_cache_get(k_idx)
    if cache_data is None:
        cache_data = _load_matrix_entry_from_disk(k_idx)
        _cpu_cache_put(k_idx, cache_data, allow_evict=True)
    if use_original:
        idx_cpu = cache_data['orig_indices']
        val_cpu = cache_data['orig_values']
        b_cpu = cache_data['orig_b']
    else:
        idx_cpu = cache_data['indices']
        val_cpu = cache_data['values']
        b_cpu = cache_data['b']
    shape = cache_data['shape']
    if dtype is not None:
        val_cpu = val_cpu.to(dtype)
        b_cpu = b_cpu.to(dtype)
    if device.type == 'cpu':
        return (torch.sparse_coo_tensor(idx_cpu, val_cpu, shape, device=device), b_cpu)
    cache_key = _device_cache_key(k_idx, device, dtype, use_original)
    cached = _gpu_cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        idx_gpu = idx_cpu.to(device, non_blocking=True)
        val_gpu = val_cpu.to(device, non_blocking=True)
        A_gpu = torch.sparse_coo_tensor(idx_gpu, val_gpu, shape, device=device).coalesce()
        b_gpu = b_cpu.to(device, non_blocking=True)
        _gpu_cache_put(cache_key, A_gpu, b_gpu)
        return (A_gpu, b_gpu)
    except Exception as e:
        if 'out of memory' in str(e).lower():
            clear_gpu_matrix_cache()
        idx_gpu = idx_cpu.to(device)
        val_gpu = val_cpu.to(device)
        A_gpu = torch.sparse_coo_tensor(idx_gpu, val_gpu, shape, device=device).coalesce()
        b_gpu = b_cpu.to(device)
        _gpu_cache_put(cache_key, A_gpu, b_gpu)
        return (A_gpu, b_gpu)

def load_matrix_to_cache(data_mapping, n_total, device_ids, is_main_process_flag=True, rank=0, world_size=1):
    global MATRIX_DATA_MAPPING, MATRIX_N_TOTAL
    MATRIX_DATA_MAPPING = data_mapping
    MATRIX_N_TOTAL = n_total
    valid_indices = [k for k in range(1, n_total + 1) if k in data_mapping]
    rank_indices = valid_indices[int(rank)::max(int(world_size), 1)]
    if CPU_MATRIX_CACHE_LIMIT_BYTES <= 0:
        return
    loaded = 0
    skipped = 0
    for k_idx in rank_indices:
        try:
            entry = _load_matrix_entry_from_disk(k_idx, data_mapping)
            cached = _cpu_cache_put(k_idx, entry, allow_evict=False)
            if not cached:
                skipped += 1
                if MATRIX_CACHE_BYTES > 0:
                    break
                continue
            loaded += 1
            if is_main_process_flag and loaded % 100 == 0:
                cache_mb = MATRIX_CACHE_BYTES / (1024 * 1024)
        except Exception as e:
            skipped += 1
    if is_main_process_flag:
        cache_mb = MATRIX_CACHE_BYTES / (1024 * 1024)

def save_training_data(training_data, save_path='training_data.json'):
    if not is_main_process():
        return
    import json

    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for (k, v) in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj
        else:
            return str(obj)
    serializable_data = convert_to_serializable(training_data)
    with open(save_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)

def load_pretrained_model(model_dir, solver, n_iter, device, is_main_process):
    main_process = is_main_process() if callable(is_main_process) else bool(is_main_process)
    if main_process:
        print(f'📂 正在加载预训练模型权重: {model_dir}')
    raw_model = solver.module if isinstance(solver, (DDP,)) else solver
    use_shared = getattr(raw_model, 'use_shared_real_imag', False)
    all_exist = True
    for i in range(n_iter):
        shared_path = os.path.join(model_dir, f'shared_iter_{i}.pth')
        if not os.path.exists(shared_path):
            all_exist = False
            if main_process:
                print(f'⚠️  模型文件不存在: {shared_path}')
            break
        continue
    if not all_exist:
        if main_process:
            print('   将从随机初始化开始训练')
        return False
    try:
        for i in range(n_iter):
            shared_path = os.path.join(model_dir, f'shared_iter_{i}.pth')
            shared_net = raw_model.model_complex.get_network(i)
            shared_net.load_state_dict(torch.load(shared_path, map_location=device, weights_only=True))
            continue
        if main_process:
            print(f'✅ 预训练模型权重加载成功!')
            print(f'   已加载 {n_iter} 个迭代网络的权重')
            print(f'   ⚠️  注意：优化器、学习率调度器等状态已重新初始化')
        return True
    except Exception as e:
        if main_process:
            print(f'❌ 加载预训练模型失败: {e}')
            print('   将从随机初始化开始训练')
        return False

def save_solver_iter_weights(raw_model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    n_iter = raw_model.n_iter
    for i in range(n_iter):
        torch.save(raw_model.model_complex.get_network(i).state_dict(), os.path.join(save_dir, f'shared_iter_{i}.pth'))

def unwrap_solver_model(solver):
    return solver.module if isinstance(solver, (DDP,)) else solver

def get_training_checkpoint_path():
    if CHECKPOINT_PATH:
        return CHECKPOINT_PATH
    return os.path.join(SAVE_DIR, 'latest_checkpoint.pt')

def save_training_checkpoint(checkpoint_path, solver, optimizer, scheduler, amp_grad_scaler, epoch, best_saved_loss, best_epoch, early_stopping_best_loss, early_stopping_counter, training_data, train_losses, test_losses, train_mse_losses, test_mse_losses, train_res_losses, test_res_losses, last_test_loss, last_test_mse, last_test_res, last_test_phi, last_validated_epoch, hybrid_loss_data=None, checkpoint_reason=None, verbose=False):
    os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
    raw_model = unwrap_solver_model(solver)
    checkpoint = {'epoch': int(epoch), 'next_epoch': int(epoch) + 1, 'total_epochs': int(TOTAL_EPOCHS), 'n_iter': int(N_ITER), 'model_state_dict': raw_model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None, 'amp_grad_scaler_state_dict': amp_grad_scaler.state_dict() if amp_grad_scaler is not None and amp_grad_scaler.is_enabled() else None, 'best_saved_loss': float(best_saved_loss), 'best_epoch': int(best_epoch), 'early_stopping_best_loss': float(early_stopping_best_loss), 'early_stopping_counter': int(early_stopping_counter), 'training_data': training_data, 'train_losses': train_losses, 'test_losses': test_losses, 'train_mse_losses': train_mse_losses, 'test_mse_losses': test_mse_losses, 'train_res_losses': train_res_losses, 'test_res_losses': test_res_losses, 'last_test_loss': float(last_test_loss), 'last_test_mse': float(last_test_mse), 'last_test_res': float(last_test_res), 'last_test_phi': float(last_test_phi), 'last_validated_epoch': last_validated_epoch, 'hybrid_loss_data': hybrid_loss_data, 'amp_enabled': bool(AMP_ENABLED), 'amp_dtype': AMP_DTYPE, 'checkpoint_reason': checkpoint_reason, 'saved_at': time.strftime('%Y-%m-%d %H:%M:%S')}
    tmp_path = checkpoint_path + '.tmp'
    torch.save(checkpoint, tmp_path)
    os.replace(tmp_path, checkpoint_path)
    log_regular_saves = os.environ.get('FEMGRL_CHECKPOINT_LOG_SAVES', '0').lower() in {'1', 'true', 'yes', 'on'}
    if verbose and (checkpoint_reason not in {'periodic', 'validation'} or log_regular_saves):
        reason_tag = f' [{checkpoint_reason}]' if checkpoint_reason else ''
        print(f'Checkpoint saved{reason_tag}: epoch={epoch}, resume_epoch={epoch + 1}, path={checkpoint_path}')

def load_training_checkpoint(checkpoint_path, solver, optimizer, scheduler, amp_grad_scaler, device, main_process):
    if not CHECKPOINT_RESUME_ENABLED or not checkpoint_path or (not os.path.exists(checkpoint_path)):
        return None
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        raw_model = unwrap_solver_model(solver)
        raw_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer_state = checkpoint.get('optimizer_state_dict')
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        scheduler_state = checkpoint.get('scheduler_state_dict')
        if scheduler is not None and scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        scaler_state = checkpoint.get('amp_grad_scaler_state_dict')
        if amp_grad_scaler is not None and amp_grad_scaler.is_enabled() and (scaler_state is not None):
            amp_grad_scaler.load_state_dict(scaler_state)
        start_epoch = int(checkpoint.get('next_epoch', int(checkpoint.get('epoch', -1)) + 1))
        start_epoch = max(0, min(start_epoch, TOTAL_EPOCHS))
        if main_process:
            print(f'Checkpoint loaded: {checkpoint_path}')
            print(f'   resume from epoch {start_epoch}')
        return {'start_epoch': start_epoch, 'best_saved_loss': checkpoint.get('best_saved_loss', float('inf')), 'best_epoch': checkpoint.get('best_epoch', -1), 'early_stopping_best_loss': checkpoint.get('early_stopping_best_loss', float('inf')), 'early_stopping_counter': checkpoint.get('early_stopping_counter', 0), 'training_data': checkpoint.get('training_data'), 'train_losses': checkpoint.get('train_losses'), 'test_losses': checkpoint.get('test_losses'), 'train_mse_losses': checkpoint.get('train_mse_losses'), 'test_mse_losses': checkpoint.get('test_mse_losses'), 'train_res_losses': checkpoint.get('train_res_losses'), 'test_res_losses': checkpoint.get('test_res_losses'), 'last_test_loss': checkpoint.get('last_test_loss', float('nan')), 'last_test_mse': checkpoint.get('last_test_mse', float('nan')), 'last_test_res': checkpoint.get('last_test_res', float('nan')), 'last_test_phi': checkpoint.get('last_test_phi', float('nan')), 'last_validated_epoch': checkpoint.get('last_validated_epoch'), 'hybrid_loss_data': checkpoint.get('hybrid_loss_data')}
    except Exception as exc:
        if main_process:
            print(f'Warning: checkpoint resume failed from {checkpoint_path}: {exc}')
            print('   training will continue from the configured initial state')
        return None

class PhiSAGESolver(nn.Module):

    def __init__(self, input_feats, output_feats, n_iter=N_ITER):
        super(PhiSAGESolver, self).__init__()
        self.n_iter = n_iter
        self.use_shared_real_imag = NETWORK_SHARED_REAL_IMAG_HEADS
        self.model_complex = BuildGCNSharedHeadsList(input_feats, n_iter)

    def get_precision_module(self):
        return self.model_complex

    def predict_delta(self, batch, iter_idx):
        delta = self.model_complex(batch, iter_idx)
        return (delta[:, 0], delta[:, 1])

    def get_batch_train_matrix(self, batch, device, matrix_dtype):
        A_train = getattr(batch, 'A_train', None)
        b_train = getattr(batch, 'b_train', None)
        if A_train is None or b_train is None:
            return (None, None)
        A_train = A_train.to(device=device, dtype=matrix_dtype)
        if getattr(A_train, 'is_sparse', False):
            A_train = A_train.coalesce()
        b_train = b_train.to(device=device, dtype=matrix_dtype).view(-1)
        return (A_train, b_train)

    def compute_residual_features(self, E_real_cur, E_imag_cur, node_batch, B, A_list, b_list, A_batch, b_batch, data_dtype):
        with torch.no_grad():
            if A_batch is not None and b_batch is not None:
                E_c = torch.complex(E_real_cur, E_imag_cur)
                Ax = torch.sparse.mm(A_batch, E_c.unsqueeze(-1)).squeeze(-1)
                r_c = b_batch - Ax
                r_real = r_c.real.to(dtype=data_dtype)
                r_imag = r_c.imag.to(dtype=data_dtype)
                del E_c, Ax, r_c
                return (r_real, r_imag)
            r_real_list = []
            r_imag_list = []
            for b_idx in range(B):
                mask = node_batch == b_idx
                E_r = E_real_cur[mask]
                E_i = E_imag_cur[mask]
                A = A_list[b_idx]
                b_vec = b_list[b_idx]
                E_c = torch.complex(E_r, E_i)
                Ax = torch.sparse.mm(A, E_c.unsqueeze(-1)).squeeze(-1)
                r_c = b_vec - Ax
                r_real_list.append(r_c.real.to(dtype=data_dtype))
                r_imag_list.append(r_c.imag.to(dtype=data_dtype))
                del E_c, Ax, r_c
            r_real = torch.cat(r_real_list, dim=0)
            r_imag = torch.cat(r_imag_list, dim=0)
            del r_real_list, r_imag_list
            return (r_real, r_imag)

    def forward(self, data_list, epoch=None):
        batch = data_list
        device = batch.x.device
        try:
            model_device = next(self.parameters()).device
            if device != model_device:
                batch = batch.to(model_device)
                device = model_device
        except (StopIteration, RuntimeError):
            pass
        k_all = batch.k_idx
        node_batch = batch.batch
        B = k_all.size(0)
        is_double = get_model_parameter_dtype(self.get_precision_module()) == torch.float64
        if is_double:
            eps_feat = batch.x[:, 0:2].double()
            bg_real = batch.x[:, 6].double()
            bg_imag = batch.x[:, 7].double()
            current_E_real = batch.x[:, 4].double()
            current_E_imag = batch.x[:, 5].double()
        else:
            eps_feat = batch.x[:, 0:2].float()
            bg_real = batch.x[:, 6].float()
            bg_imag = batch.x[:, 7].float()
            current_E_real = batch.x[:, 4].float()
            current_E_imag = batch.x[:, 5].float()
        precond_scale_value = PRECOND_TRAIN_SCALE if hasattr(batch, 'recovery_operator') else 1.0
        precond_scale = torch.tensor(precond_scale_value, dtype=current_E_real.dtype, device=device)
        if precond_scale_value != 1.0:
            current_E_real = current_E_real * precond_scale
            current_E_imag = current_E_imag * precond_scale
            bg_real = bg_real * precond_scale
            bg_imag = bg_imag * precond_scale
        matrix_dtype = torch.complex128 if is_double else torch.complex64
        (A_batch, b_batch) = self.get_batch_train_matrix(batch, device, matrix_dtype)
        A_list = []
        b_list = []
        if A_batch is None:
            if GRAPH_CACHE_ENABLED and GRAPH_CACHE_STRICT:
                raise RuntimeError('GRAPH_CACHE_STRICT=1 requires A_train/b_train inside every graph cache file. Please rerun precompute_graph_cache.py with the updated code.')
            for b_idx in range(B):
                k = int(k_all[b_idx].item())
                (A, b) = get_Ab(k, device, matrix_dtype, use_original=False)
                A_list.append(A)
                b_list.append(b)
        E_real_cur = current_E_real
        E_imag_cur = current_E_imag
        for iter_idx in range(self.n_iter):
            (r_real, r_imag) = self.compute_residual_features(E_real_cur, E_imag_cur, node_batch, B, A_list, b_list, A_batch, b_batch, torch.float64 if is_double else torch.float32)
            x_in = torch.cat([eps_feat, r_real.view(-1, 1), r_imag.view(-1, 1), E_real_cur.view(-1, 1), E_imag_cur.view(-1, 1), bg_real.view(-1, 1), bg_imag.view(-1, 1)], dim=1)
            batch.x = x_in
            delta = None
            (delta_real, delta_imag) = self.predict_delta(batch, iter_idx)
            delta_real = delta_real.view(-1)
            delta_imag = delta_imag.view(-1)
            E_real_next = E_real_cur + delta_real
            E_imag_next = E_imag_cur + delta_imag
            del x_in, delta, delta_real, delta_imag, r_real, r_imag
            E_real_cur = E_real_next
            E_imag_cur = E_imag_next
        if hasattr(batch, 'recovery_operator'):
            R = batch.recovery_operator
            rec_type = batch.recovery_type[0] if isinstance(batch.recovery_type, list) else batch.recovery_type
            if rec_type == 'vector_diagonal':
                R_c = R.to(device=device, dtype=matrix_dtype).view(-1)
                x_true_c = torch.complex(batch.y[:, 0].to(device=device, dtype=E_real_cur.dtype), batch.y[:, 1].to(device=device, dtype=E_imag_cur.dtype)).to(matrix_dtype)
                z_true_c = x_true_c / R_c * precond_scale.to(dtype=matrix_dtype)
                z_true_real = z_true_c.real.to(E_real_cur.dtype)
                z_true_imag = z_true_c.imag.to(E_imag_cur.dtype)
            else:
                raise NotImplementedError('稀疏矩阵的预处理真解推导计算成本过高，请先使用对角缩放。')
        else:
            z_true_real = batch.y[:, 0] * precond_scale
            z_true_imag = batch.y[:, 1] * precond_scale
        batch_z_true = torch.stack([z_true_real, z_true_imag], dim=1)
        num_nodes = batch.x.size(0)
        loss_sum = compute_mse_loss(E_real_cur, E_imag_cur, batch_z_true, num_nodes)
        phi_loss_sum = torch.tensor(0.0, device=device)
        res_loss_sum = torch.tensor(0.0, device=device, dtype=loss_sum.dtype)
        phys_mse_sum = torch.tensor(0.0, device=device, dtype=loss_sum.dtype)
        with torch.no_grad():
            if hasattr(batch, 'recovery_operator'):
                if rec_type == 'vector_diagonal':
                    if 'R_c' not in locals():
                        R_c = R.to(device=device, dtype=matrix_dtype).view(-1)
                    z_pred_c = torch.complex(E_real_cur / precond_scale, E_imag_cur / precond_scale).to(matrix_dtype)
                    x_pred_c = R_c * z_pred_c
                    x_real_pred = x_pred_c.real.to(E_real_cur.dtype)
                    x_imag_pred = x_pred_c.imag.to(E_imag_cur.dtype)
                else:
                    E_c = torch.complex(E_real_cur / precond_scale, E_imag_cur / precond_scale)
                    R = R.to(device=device, dtype=matrix_dtype)
                    if getattr(R, 'is_sparse', False):
                        R = R.coalesce()
                        x_pred = torch.sparse.mm(R, E_c.unsqueeze(-1)).squeeze(-1)
                    else:
                        x_pred = R @ E_c
                    x_real_pred = x_pred.real.to(E_real_cur.dtype)
                    x_imag_pred = x_pred.imag.to(E_imag_cur.dtype)
            else:
                x_real_pred = E_real_cur / precond_scale
                x_imag_pred = E_imag_cur / precond_scale
            true_real = batch.y[:, 0]
            true_imag = batch.y[:, 1]
            phys_mse_sum = 0.5 * ((x_real_pred - true_real).pow(2).sum() + (x_imag_pred - true_imag).pow(2).sum())
            for b_idx in range(B):
                mask = node_batch == b_idx
                pred_real = x_real_pred[mask]
                pred_imag = x_imag_pred[mask]
                true_r = true_real[mask]
                true_i = true_imag[mask]
                numerator = torch.abs(true_r - pred_real).pow(2).sum() + torch.abs(true_i - pred_imag).pow(2).sum()
                denominator = torch.abs(true_r).pow(2).sum() + torch.abs(true_i).pow(2).sum()
                if denominator > 1e-10:
                    res_loss_sum = res_loss_sum + numerator / denominator
                else:
                    res_loss_sum = res_loss_sum + numerator
                del pred_real, pred_imag, true_r, true_i
        del A_list, b_list
        if A_batch is not None:
            del A_batch, b_batch
        del E_real_cur, E_imag_cur, true_real, true_imag
        num_nodes = batch.x.size(0)
        num_samples = B
        num_nodes_tensor = torch.tensor(num_nodes, dtype=torch.float32, device=device)
        num_samples_tensor = torch.tensor(num_samples, dtype=torch.float32, device=device)
        mse_loss_tensor = phys_mse_sum.to(dtype=torch.float32, device=device)
        phi_loss_tensor = phi_loss_sum.to(dtype=torch.float32, device=device) if 'phi_loss_sum' in locals() else torch.tensor(0.0, dtype=torch.float32, device=device)
        return torch.stack([loss_sum.to(dtype=torch.float32), res_loss_sum.to(dtype=torch.float32), num_nodes_tensor, num_samples_tensor, mse_loss_tensor, phi_loss_tensor])

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = MASTER_PORT
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['NCCL_TIMEOUT'] = '1800000'
    os.environ['NCCL_IB_DISABLE'] = '1'
    os.environ['NCCL_SOCKET_IFNAME'] = 'lo'
    os.environ['NCCL_DEBUG'] = 'WARN'
    dist.init_process_group('nccl', rank=rank, world_size=world_size, timeout=timedelta(minutes=30))
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    return device

def cleanup_ddp():
    dist.destroy_process_group()

def compute_mse_loss(real, imag, target, num_nodes=None):
    return 0.5 * ((real - target[:, 0]).square() + (imag - target[:, 1]).square()).sum()

def extract_loss_and_num_nodes(outputs):
    if outputs.ndim != 1 or outputs.numel() != 6:
        raise ValueError('Expected one six-element DDP metrics tensor')
    return (outputs[0], outputs[1], int(outputs[2].item()), int(outputs[3].item()), outputs[4], outputs[5])

def main_worker(rank, world_size):
    device = setup_ddp(rank, world_size)
    try:
        train_worker(rank, world_size, device)
    finally:
        if dist.is_initialized():
            cleanup_ddp()

def train_worker(rank, world_size, device):
    set_global_seed(RANDOM_SEED)
    primary = rank == 0
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if primary:
        sys.stdout = Logger(os.path.join(OUTPUT_DIR, 'training_log.txt'))
        sys.stderr = sys.stdout
    (mapping, n_total) = scan_all_data(root_data_path)
    excluded = load_excluded_k_indices(get_runtime_exclude_k_file())
    eligible = [k for k in range(1, n_total + 1) if k in mapping and k not in excluded]
    if not eligible:
        raise RuntimeError('No eligible training samples')
    (train_k, test_k) = train_test_split(eligible, test_size=0.2, random_state=RANDOM_SEED)
    (train_k, test_k, manifest) = load_training_subset(train_k, test_k, os.environ.get('FEMGRL_TRAIN_SUBSET_FILE'), mapping)
    selected_mapping = filter_data_mapping_by_k(mapping, train_k + test_k)
    if manifest is not None:
        target = os.path.join(OUTPUT_DIR, 'small_sample_manifest.json')
        if os.path.exists(target):
            with open(target, encoding='utf-8') as stream:
                if json.load(stream) != manifest:
                    raise ValueError('Output directory contains a different training subset')
        if primary:
            with open(target + '.tmp', 'w', encoding='utf-8') as stream:
                json.dump(manifest, stream, indent=2)
            os.replace(target + '.tmp', target)
    if primary:
        with open(os.path.join(OUTPUT_DIR, 'test_set_indices.txt'), 'w', encoding='utf-8') as stream:
            stream.write(', '.join(map(str, sorted(test_k))) + '\n')
    (train_ds, test_ds) = (KIndexGraphDataset(train_k), KIndexGraphDataset(test_k))
    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=RANDOM_SEED)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False, seed=RANDOM_SEED)
    train_loader = DataLoader(train_ds, batch_size=TOTAL_BATCH_SIZE, sampler=train_sampler, drop_last=True, pin_memory=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=TOTAL_BATCH_SIZE, sampler=test_sampler, pin_memory=False, num_workers=0)
    global MATRIX_DATA_MAPPING, MATRIX_N_TOTAL
    (MATRIX_DATA_MAPPING, MATRIX_N_TOTAL) = (selected_mapping, n_total)
    if not (GRAPH_CACHE_ENABLED and GRAPH_CACHE_STRICT):
        load_matrix_to_cache(selected_mapping, n_total, [rank], primary, rank=rank, world_size=world_size)
    dist.barrier()
    solver = PhiSAGESolver(input_feats=8, output_feats=1, n_iter=N_ITER).float().to(device)
    solver = DDP(solver, device_ids=[rank], output_device=rank, find_unused_parameters=False)
    optimizer = optim.AdamW(solver.parameters(), lr=LR, weight_decay=1e-05)
    scaler = create_amp_grad_scaler(device)
    patience = max(1, (REDUCE_LR_PATIENCE + VALIDATE_EVERY - 1) // VALIDATE_EVERY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=REDUCE_LR_FACTOR, patience=patience, min_lr=REDUCE_LR_MIN_LR, threshold=EARLY_STOPPING_MIN_DELTA, threshold_mode='abs')
    history = {key: [] for key in ('epochs', 'train_losses', 'test_losses', 'train_mse_losses', 'test_mse_losses', 'train_res_losses', 'test_res_losses', 'validation_epochs')}
    history['frequency_modulation'] = {'enabled': NETWORK_USE_FREQUENCY_MODULATION, 'norm_min': FREQUENCY_NORM_MIN, 'norm_max': FREQUENCY_NORM_MAX, 'embed_dim': FREQUENCY_EMBED_DIM, 'film_strength': FREQUENCY_FILM_STRENGTH}
    history['mixed_precision'] = {'enabled': AMP_ENABLED, 'dtype': AMP_DTYPE, 'grad_scaler': AMP_USE_GRAD_SCALER and AMP_DTYPE == 'fp16'}
    (start_epoch, best_epoch, stale) = (0, -1, 0)
    best_saved = early_best = float('inf')
    last_test = [float('nan')] * 3
    last_validated = None
    checkpoint_path = get_training_checkpoint_path()
    if LOAD_PRETRAINED_MODEL:
        load_pretrained_model(PRETRAINED_MODEL_DIR or SAVE_DIR, solver, N_ITER, device, primary)
    resume = CHECKPOINT_RESUME_ENABLED and (not LOAD_PRETRAINED_MODEL or CHECKPOINT_RESUME_WITH_PRETRAIN)
    if resume:
        state = load_training_checkpoint(checkpoint_path, solver, optimizer, scheduler, scaler, device, primary)
        dist.barrier()
        if state is not None:
            start_epoch = state['start_epoch']
            (best_saved, best_epoch) = (float(state['best_saved_loss']), int(state['best_epoch']))
            (early_best, stale) = (float(state['early_stopping_best_loss']), int(state['early_stopping_counter']))
            if isinstance(state.get('training_data'), dict):
                history.update(state['training_data'])
            for key in ('train_losses', 'test_losses', 'train_mse_losses', 'test_mse_losses', 'train_res_losses', 'test_res_losses'):
                history[key] = list(state.get(key) or history.get(key, []))
            last_test = [float(state.get(key, float('nan'))) for key in ('last_test_loss', 'last_test_mse', 'last_test_res')]
            for (i, key) in enumerate(('test_losses', 'test_mse_losses', 'test_res_losses')):
                if history[key]:
                    last_test[i] = history[key][-1]
            last_validated = state.get('last_validated_epoch')
            if history['validation_epochs']:
                last_validated = history['validation_epochs'][-1]

    def save_checkpoint(epoch, reason):
        if primary:
            save_training_checkpoint(checkpoint_path, solver, optimizer, scheduler, scaler, epoch, best_saved, best_epoch, early_best, stale, history, history['train_losses'], history['test_losses'], history['train_mse_losses'], history['test_mse_losses'], history['train_res_losses'], history['test_res_losses'], *last_test, 0.0, last_validated, None, reason, CHECKPOINT_VERBOSE)
        dist.barrier()
    for epoch in range(start_epoch, TOTAL_EPOCHS):
        train_sampler.set_epoch(epoch)
        solver.train()
        optimizer.zero_grad(set_to_none=True)
        train_sum = np.zeros(5, dtype=np.float64)
        for (step, batch) in enumerate(train_loader):
            update = (step + 1) % 2 == 0 or step + 1 == len(train_loader)
            with nullcontext() if update else solver.no_sync():
                with amp_autocast_context(device):
                    (loss, rel, nodes, samples, mse, _) = extract_loss_and_num_nodes(solver(batch, epoch))
                    if nodes == 0:
                        continue
                    loss_scaled = loss / nodes / 2
                if scaler.is_enabled():
                    scaler.scale(loss_scaled).backward()
                else:
                    loss_scaled.backward()
            if update:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                if GRAD_CLIP_NORM > 0:
                    torch.nn.utils.clip_grad_norm_(solver.parameters(), GRAD_CLIP_NORM)
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            train_sum += [loss.item(), mse.item(), rel.item(), nodes, samples]
            del loss, rel, mse, loss_scaled, batch
        if epoch % 10 == 0:
            torch.cuda.empty_cache()
        train_metrics = [train_sum[0] / train_sum[3] if train_sum[3] else 0.0, train_sum[1] / train_sum[3] if train_sum[3] else 0.0, train_sum[2] / train_sum[4] if train_sum[4] else 0.0]
        validate = epoch == 0 or epoch == TOTAL_EPOCHS - 1 or epoch % VALIDATE_EVERY == 0
        if validate:
            solver.eval()
            totals = torch.zeros(5, dtype=torch.float64, device=device)
            with torch.no_grad():
                for batch in test_loader:
                    with amp_autocast_context(device):
                        (loss, rel, nodes, samples, mse, _) = extract_loss_and_num_nodes(solver(batch, epoch))
                    totals += torch.tensor([loss.item(), mse.item(), rel.item(), nodes, samples], dtype=torch.float64, device=device)
                    del loss, rel, mse, batch
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            sums = totals.tolist()
            last_test = [sums[0] / sums[3] if sums[3] else 0.0, sums[1] / sums[3] if sums[3] else 0.0, sums[2] / sums[4] if sums[4] else 0.0]
            last_validated = epoch
            history['validation_epochs'].append(epoch)
            scheduler.step(last_test[1])
            if last_test[1] < best_saved:
                (best_saved, best_epoch) = (last_test[1], epoch)
                if primary:
                    save_solver_iter_weights(unwrap_solver_model(solver), SAVE_DIR)
            torch.cuda.empty_cache()
            gc.collect()
        history['epochs'].append(epoch)
        for (key, value) in zip(('train_losses', 'train_mse_losses', 'train_res_losses'), train_metrics):
            history[key].append(value)
        for (key, value) in zip(('test_losses', 'test_mse_losses', 'test_res_losses'), last_test):
            history[key].append(value)
        metric = last_test[1] if validate else early_best
        stop = False
        if EARLY_STOPPING_ENABLED and validate and (epoch >= EARLY_STOPPING_START_EPOCH):
            if metric < early_best - EARLY_STOPPING_MIN_DELTA:
                (early_best, stale) = (metric, 0)
            else:
                stale += VALIDATE_EVERY
            stop = stale >= EARLY_STOPPING_PATIENCE
        elif metric < early_best:
            early_best = metric
        due = validate or epoch == TOTAL_EPOCHS - 1 or (epoch + 1) % CHECKPOINT_EVERY_EPOCHS == 0
        if CHECKPOINT_SAVE_ENABLED and (due or stop):
            reason = 'early_stop' if stop else 'final' if epoch == TOTAL_EPOCHS - 1 else 'validation' if validate else 'periodic'
            save_checkpoint(epoch, reason)
        if primary and (epoch % EPOCH_PRINT == 0 or stop):
            print(f'Epoch {epoch} | train MSE={train_metrics[1]:.6e} | test MSE={last_test[1]:.6e} | test relative error={last_test[2]:.6e}')
        if stop:
            break
    dist.barrier()
    if primary:
        save_training_data(history, os.path.join(OUTPUT_DIR, 'training_data.json'))
        print(f'Best epoch={best_epoch}, test MSE={best_saved:.6e}')

def main():
    if not torch.cuda.is_available():
        raise RuntimeError('DDP training requires CUDA')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'training_log.txt'), 'w', encoding='utf-8') as stream:
        stream.write('AMG edge-attention DDP training\n')
    mp.spawn(main_worker, args=(torch.cuda.device_count(),), nprocs=torch.cuda.device_count(), join=True)
if __name__ == '__main__':
    main()
