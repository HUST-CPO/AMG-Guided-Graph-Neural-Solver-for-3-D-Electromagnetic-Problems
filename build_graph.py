from training.paths import project_path
import os
import re
import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from training.config import DATASET_TYPE, DATA_ROOT_PATH, SCA_PREFIX, DATASET_DIRS_PATTERN, NETWORK_USE_GEOMETRIC_EDGE_FEATURES, EDGE_ALGEBRAIC_FEATURE_DIM, EDGE_FEATURE_MATRIX, GRAPH_CACHE_DIR, GRAPH_CACHE_VERSION, PRECOND_METHOD, PRECOND_TRAIN_SCALE, NETWORK_USE_FREQUENCY_MODULATION, FREQUENCY_FILE_PREFIX
from training.data_io import load_prefixed_txt, sample_file_ids as data_sample_file_ids, sample_folder_specs as data_sample_folder_specs, unpack_sample_info
root_data_path = str(project_path(DATA_ROOT_PATH))
sca_prefix = SCA_PREFIX
dataset_dirs_pattern = DATASET_DIRS_PATTERN
AMG_OPERATOR_FILENAME_MODE = os.environ.get('FEMGRL_AMG_OPERATOR_FILENAME_MODE', 'legacy').strip().lower()
AMG_REQUIRE_METHOD_SPECIFIC = os.environ.get('FEMGRL_AMG_REQUIRE_METHOD_SPECIFIC', '0').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}

def is_main_process():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    rank = os.environ.get('RANK')
    if rank is not None:
        return int(rank) == 0
    return True
data_mapping = {}
n_total = 0

def _graph_cache_config_tag():
    precond = PRECOND_METHOD if PRECOND_METHOD else 'raw'
    geom = 'geom1' if NETWORK_USE_GEOMETRIC_EDGE_FEATURES else 'geom0'
    freq = '_freq1' if NETWORK_USE_FREQUENCY_MODULATION else ''
    return f'{DATASET_TYPE}_{precond}_edge-{EDGE_FEATURE_MATRIX}_alg-{EDGE_ALGEBRAIC_FEATURE_DIM}_{geom}{freq}_{GRAPH_CACHE_VERSION}'

def get_graph_cache_dir(cache_dir=None):
    if cache_dir:
        return cache_dir
    if GRAPH_CACHE_DIR:
        return GRAPH_CACHE_DIR
    return os.path.join(root_data_path, 'pyg_graph_cache', _graph_cache_config_tag())

def _sanitize_cache_component(value):
    return re.sub('[^A-Za-z0-9_.-]+', '_', str(value)).strip('_')

def get_graph_cache_path(k, cache_dir=None):
    if not data_mapping or k not in data_mapping:
        scan_all_data(root_data_path)
    if k not in data_mapping:
        raise ValueError(f'k_idx {k} is not in data_mapping')
    (folder_path, name_prefix, folder_num, data_id) = unpack_sample_info(data_mapping[k], sca_prefix)
    rel_folder = os.path.relpath(folder_path, root_data_path)
    rel_slug = _sanitize_cache_component(rel_folder)
    if rel_slug in {'', '.'}:
        rel_slug = 'root'
    sample_slug = _sanitize_cache_component(f'{name_prefix}{folder_num}_{data_id}')
    filename = f'{rel_slug}__{sample_slug}.pt'
    return os.path.join(get_graph_cache_dir(cache_dir), filename)

def get_amg_operator_file(folder_path, name_prefix, folder_num, data_id):
    legacy_file = os.path.join(folder_path, f'amg_operators_{name_prefix}{folder_num}_{data_id}.pt')
    method = PRECOND_METHOD if PRECOND_METHOD else 'raw'
    method_file = os.path.join(folder_path, f'amg_operators_{method}_{name_prefix}{folder_num}_{data_id}.pt')
    if AMG_OPERATOR_FILENAME_MODE in {'method', 'method_specific', 'specific'}:
        return method_file
    if AMG_OPERATOR_FILENAME_MODE == 'auto':
        if os.path.exists(method_file) or AMG_REQUIRE_METHOD_SPECIFIC:
            return method_file
        return legacy_file
    if AMG_OPERATOR_FILENAME_MODE != 'legacy':
        raise ValueError('FEMGRL_AMG_OPERATOR_FILENAME_MODE must be one of: legacy, auto, method')
    return legacy_file

def _torch_load_full(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')

def _torch_load_weights(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')

def _scipy_to_torch_sparse_complex(matrix):
    coo = matrix.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.complex64)
    return torch.sparse_coo_tensor(indices, values, coo.shape).coalesce()

def _as_float_scale(value, default=1.0):
    if value is None:
        return float(default)
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return float(default)
        return float(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if not value:
            return float(default)
        return _as_float_scale(value[0], default)
    return float(value)

def _ensure_training_b_scale(data):
    if not hasattr(data, 'b_train'):
        return data
    target_scale = float(PRECOND_TRAIN_SCALE) if hasattr(data, 'recovery_operator') else 1.0
    current_scale = _as_float_scale(getattr(data, 'b_train_scale', None), 1.0)
    if abs(current_scale - target_scale) > 1e-12:
        data.b_train = data.b_train * (target_scale / current_scale)
        data.b_train_scale = torch.tensor([target_scale], dtype=torch.float32)
    return data

def _load_sample_frequency(folder_path, name_prefix, folder_num, data_id):
    freq_data = load_prefixed_txt(folder_path, FREQUENCY_FILE_PREFIX, folder_num, data_id, name_prefix, dtype=np.float64)
    if freq_data.size == 0:
        raise ValueError(f'Empty frequency file for sample {name_prefix}{folder_num}_{data_id}')
    freq_hz = float(freq_data.reshape(-1)[0])
    return torch.tensor([[freq_hz]], dtype=torch.float32)

def _ensure_frequency_field(k, data):
    if not NETWORK_USE_FREQUENCY_MODULATION:
        return data
    if hasattr(data, 'freq'):
        freq = torch.as_tensor(data.freq, dtype=torch.float32).reshape(1, -1)
        if freq.size(1) != 1:
            freq = freq[:, :1]
        data.freq = freq
        return data
    if not data_mapping or k not in data_mapping:
        scan_all_data(root_data_path)
    if k not in data_mapping:
        raise ValueError(f'k_idx {k} is not in data_mapping')
    (folder_path, name_prefix, folder_num, data_id) = unpack_sample_info(data_mapping[k], sca_prefix)
    data.freq = _load_sample_frequency(folder_path, name_prefix, folder_num, data_id)
    return data

def _cached_graph_is_valid(k, data, cache_path):
    if PRECOND_METHOD and (not hasattr(data, 'recovery_operator')):
        if is_main_process():
            print(f'Warning: PyG graph cache missing recovery_operator for k_idx={k}: {cache_path}; rebuilding from preconditioned files.')
        return False
    if PRECOND_METHOD and (not hasattr(data, 'recovery_type')):
        if is_main_process():
            print(f'Warning: PyG graph cache missing recovery_type for k_idx={k}: {cache_path}; rebuilding from preconditioned files.')
        return False
    return True

def load_graph_data_cache(k, cache_dir=None):
    cache_path = get_graph_cache_path(k, cache_dir)
    if not os.path.exists(cache_path):
        return None
    data = _torch_load_full(cache_path)
    if not _cached_graph_is_valid(k, data, cache_path):
        return None
    data.k_idx = torch.tensor([int(k)])
    data = _ensure_training_b_scale(data)
    data = _ensure_frequency_field(int(k), data)
    return data

def save_graph_data_cache(k, data=None, cache_dir=None, overwrite=False):
    cache_path = get_graph_cache_path(k, cache_dir)
    if os.path.exists(cache_path) and (not overwrite):
        return (cache_path, False)
    if data is None:
        data = build_graph_data(k)
    data.k_idx = torch.tensor([int(k)])
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    tmp_path = f'{cache_path}.tmp.{os.getpid()}'
    torch.save(data, tmp_path)
    os.replace(tmp_path, cache_path)
    return (cache_path, True)

def build_or_load_graph_data(k, use_cache=True, cache_dir=None, write_missing=False, strict_cache=False):
    if use_cache:
        data = load_graph_data_cache(k, cache_dir)
        if data is not None:
            return data
        if strict_cache:
            raise FileNotFoundError(f'PyG graph cache is missing for k_idx={k}: {get_graph_cache_path(k, cache_dir)}')
    data = build_graph_data(k)
    if use_cache and write_missing:
        save_graph_data_cache(k, data=data, cache_dir=cache_dir, overwrite=True)
    return data

def _is_torch_sparse_tensor(value):
    if not isinstance(value, torch.Tensor):
        return False
    if getattr(value, 'is_sparse', False):
        return True
    if getattr(value, 'is_sparse_csr', False):
        return True
    return str(getattr(value, 'layout', '')).startswith('torch.sparse_')

def _torch_matrix_to_scipy_csr_complex(value):
    if not isinstance(value, torch.Tensor):
        raise TypeError(f'A_new must be a torch.Tensor, got {type(value)}')
    value = value.detach().cpu()
    if _is_torch_sparse_tensor(value):
        if not getattr(value, 'is_sparse', False):
            value = value.to_sparse_coo()
        value = value.coalesce()
        indices = value.indices().numpy()
        values = value.values().numpy()
        return sp.coo_matrix((values, (indices[0], indices[1])), shape=tuple(value.shape)).tocsr()
    return sp.csr_matrix(value.numpy())

def _require_rows(name, array, expected_rows):
    if array.shape[0] != expected_rows:
        raise ValueError(f'{name} row count mismatch: got {array.shape[0]}, expected {expected_rows}')

def _require_feature_dim(name, array, expected_dim):
    if array.ndim != 2 or array.shape[1] != expected_dim:
        raise ValueError(f'{name} feature dim mismatch: got {tuple(array.shape)}, expected (*, {expected_dim})')

class AMGData(Data):

    @staticmethod
    def _amg_level_from_edge_index_key(key):
        match = re.fullmatch('edge_index_(\\d+)', str(key))
        return int(match.group(1)) if match else None

    def _num_nodes_at_level(self, level, value=None):
        attr_name = f'num_nodes_level_{level}'
        if hasattr(self, attr_name):
            level_nodes = getattr(self, attr_name)
            if isinstance(level_nodes, torch.Tensor):
                return int(level_nodes.reshape(-1)[0].item())
            return int(level_nodes)
        if level == 0 and self.num_nodes is not None:
            return int(self.num_nodes)
        if value is not None and isinstance(value, torch.Tensor) and (value.numel() > 0):
            return int(value.max().item()) + 1
        return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if re.fullmatch('[RP]_\\d+', str(key)):
            return (0, 1)
        if key in {'A_train', 'A_original'} and _is_torch_sparse_tensor(value):
            return (0, 1)
        if key == 'recovery_operator' and _is_torch_sparse_tensor(value):
            return (0, 1)
        if self._amg_level_from_edge_index_key(key) is not None:
            return -1
        return super().__cat_dim__(key, value, *args, **kwargs)

    def __inc__(self, key, value, *args, **kwargs):
        level = self._amg_level_from_edge_index_key(key)
        if level is not None:
            return self._num_nodes_at_level(level, value)
        return super().__inc__(key, value, *args, **kwargs)

def _sample_folder_specs(folder_name):
    return data_sample_folder_specs(folder_name, sca_prefix)

def _iter_dataset_paths(root_path):
    paths = []
    root_name = os.path.basename(os.path.normpath(root_path))
    if root_name and _sample_folder_specs(root_name):
        paths.append(root_path)
    if any((os.path.isdir(os.path.join(root_path, item)) and _sample_folder_specs(item) for item in os.listdir(root_path))):
        paths.append(root_path)
    paths.extend((os.path.join(root_path, item) for item in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, item)) and (not _sample_folder_specs(item)) and item.startswith(dataset_dirs_pattern.replace('*', ''))))
    return sorted(dict.fromkeys(paths))

def _collect_sample_folders(dataset_path):
    entries = []
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    for (name_prefix, folder_num, family_order) in _sample_folder_specs(dataset_name):
        entries.append((family_order, folder_num, dataset_name, dataset_path, name_prefix))
        return sorted(entries)
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for (name_prefix, folder_num, family_order) in _sample_folder_specs(folder_name):
            entries.append((family_order, folder_num, folder_name, folder_path, name_prefix))
    return sorted(entries)

def _sample_file_ids(folder_path, name_prefix, folder_num):
    return data_sample_file_ids(folder_path, name_prefix, folder_num)

def load_file_data(folder_path, prefix, folder_num, data_id, name_prefix=None):
    return load_prefixed_txt(folder_path, prefix, folder_num, data_id, name_prefix or sca_prefix, dtype=np.float32)

def _encode_mag_phase(values):
    values = np.asarray(values)
    return np.concatenate([np.abs(values).reshape(-1, 1), np.angle(values).reshape(-1, 1)], axis=1)

def _select_edge_complex_values(rows, values, diag):
    source = EDGE_FEATURE_MATRIX.lower()
    if source == 't':
        safe_diag = diag.copy()
        safe_diag[np.abs(safe_diag) < 1e-12] = 1.0 + 0j
        return -values / safe_diag[rows]
    if source == 'a':
        return values
    raise ValueError("EDGE_FEATURE_MATRIX must be one of: 't', 'a'")

def build_edge_attr_from_A(rows, cols, values, num_nodes):
    A_mat = sp.coo_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
    diag = A_mat.diagonal().astype(np.complex128)
    mask = rows != cols
    src = rows[mask]
    dst = cols[mask]
    off_values = values[mask]
    edge_values = _select_edge_complex_values(src, off_values, diag)
    edge_index = torch.tensor(np.vstack([dst, src]), dtype=torch.long)
    edge_attr = torch.tensor(_encode_mag_phase(edge_values), dtype=torch.float32)
    (edge_index, edge_attr) = coalesce(edge_index, edge_attr, num_nodes, num_nodes)
    return (edge_index, edge_attr)

def build_edge_geometry(folder_path, name_prefix, folder_num, data_id, num_nodes):
    edge_data = load_file_data(folder_path, 'edge', folder_num, data_id, name_prefix)
    vertex_data = load_file_data(folder_path, 'vertex', folder_num, data_id, name_prefix)
    v1_idx = edge_data[:, 0].astype(int) - 1
    v2_idx = edge_data[:, 1].astype(int) - 1
    edge_vec = vertex_data[v2_idx] - vertex_data[v1_idx]
    edge_length = np.linalg.norm(edge_vec, axis=1, keepdims=True).astype(np.float32)
    edge_dir = edge_vec / (edge_length + 1e-12)
    if edge_dir.shape[0] != num_nodes:
        raise ValueError(f'edge/vertex geometry size mismatch: edge={edge_dir.shape[0]}, num_nodes={num_nodes}')
    return (torch.tensor(edge_dir, dtype=torch.float32), torch.tensor(edge_length, dtype=torch.float32))

def validate_edge_attr_dim(edge_attr, name):
    if edge_attr is None:
        return
    if edge_attr.dim() != 2 or edge_attr.shape[1] != EDGE_ALGEBRAIC_FEATURE_DIM:
        raise ValueError(f'{name} dim mismatch: got {tuple(edge_attr.shape)}, expected (*, {EDGE_ALGEBRAIC_FEATURE_DIM}). Please rerun amg_preprocess.py after changing edge feature config.')

def build_graph_data(k):
    if k not in data_mapping:
        raise ValueError(f'k_idx {k} 未在映射中找到。')
    info = data_mapping[k]
    (folder_path, name_prefix, folder_num, data_id) = unpack_sample_info(info, sca_prefix)
    precond_file = os.path.join(folder_path, f'precond_{PRECOND_METHOD}_{name_prefix}{folder_num}_{data_id}.pt') if PRECOND_METHOD else None
    if precond_file and os.path.exists(precond_file):
        precond_data = _torch_load_weights(precond_file)
        A_mat = _torch_matrix_to_scipy_csr_complex(precond_data['A_new'])
        if A_mat.shape[0] != A_mat.shape[1]:
            raise ValueError(f'A_new must be square, got {A_mat.shape}')
        coo = A_mat.tocoo()
        rows = coo.row.astype(np.int64)
        cols = coo.col.astype(np.int64)
        values = coo.data
        num_nodes = A_mat.shape[0]
        b_vec = torch.as_tensor(precond_data['b_new']).detach().cpu().reshape(-1).numpy()
        Eb_new_vec = torch.as_tensor(precond_data['Eb_new']).detach().cpu().reshape(-1).numpy()
        A_train_tensor = _scipy_to_torch_sparse_complex(A_mat)
        b_train_tensor = torch.as_tensor(b_vec, dtype=torch.complex64) * float(PRECOND_TRAIN_SCALE)
        b_train_scale = float(PRECOND_TRAIN_SCALE)
        if b_vec.shape[0] != num_nodes:
            raise ValueError(f'b_new length mismatch: got {b_vec.shape[0]}, expected {num_nodes}')
        if Eb_new_vec.shape[0] != num_nodes:
            raise ValueError(f'Eb_new length mismatch: got {Eb_new_vec.shape[0]}, expected {num_nodes}')
        Eb_data = np.stack([Eb_new_vec.real, Eb_new_vec.imag], axis=1).astype(np.float32)
        Eb_vec = Eb_new_vec
        recovery_op = precond_data['recovery_operator']
        recovery_type = precond_data['recovery_type']
    else:
        if PRECOND_METHOD and is_main_process():
            print(f'Warning: 找不到预处理文件 {precond_file}, 正在回退到原始文本读取模式。')
        Aij_data = load_file_data(folder_path, 'Aij', folder_num, data_id, name_prefix)
        Av_data = load_file_data(folder_path, 'Av', folder_num, data_id, name_prefix)
        _require_feature_dim('Aij', Aij_data, 2)
        _require_feature_dim('Av', Av_data, 2)
        if Aij_data.shape[0] != Av_data.shape[0]:
            raise ValueError(f'Aij/Av row count mismatch: Aij={Aij_data.shape[0]}, Av={Av_data.shape[0]}')
        rows = Aij_data[:, 0].astype(int) - 1
        cols = Aij_data[:, 1].astype(int) - 1
        num_nodes = max(rows.max(), cols.max()) + 1
        values = Av_data[:, 0] + 1j * Av_data[:, 1]
        A_mat = sp.coo_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()
        b_data = load_file_data(folder_path, 'b', folder_num, data_id, name_prefix)
        _require_rows('b', b_data, num_nodes)
        _require_feature_dim('b', b_data, 2)
        try:
            Eb_data = load_file_data(folder_path, 'Eb', folder_num, data_id, name_prefix)
        except FileNotFoundError:
            Eb_data = b_data.copy()
        b_vec = b_data[:, 0] + 1j * b_data[:, 1]
        Eb_vec = Eb_data[:, 0] + 1j * Eb_data[:, 1]
        A_train_tensor = _scipy_to_torch_sparse_complex(A_mat)
        b_train_tensor = torch.as_tensor(b_vec, dtype=torch.complex64)
        b_train_scale = 1.0
        recovery_op = None
        recovery_type = None
    operator_file = get_amg_operator_file(folder_path, name_prefix, folder_num, data_id)
    ops = _torch_load_full(operator_file) if os.path.exists(operator_file) else None
    if ops is not None and 'edge_index_0' in ops and ('edge_attr_0' in ops):
        edge_index = ops['edge_index_0']
        edge_attr = ops['edge_attr_0']
        validate_edge_attr_dim(edge_attr, 'edge_attr_0')
        edge_attr_source = f'offline AMG operators: {os.path.basename(operator_file)}'
    else:
        (edge_index, edge_attr) = build_edge_attr_from_A(rows, cols, values, num_nodes)
        validate_edge_attr_dim(edge_attr, 'edge_attr')
        edge_attr_source = 'online fallback from A'
    edge_dir = None
    edge_length = None
    if NETWORK_USE_GEOMETRIC_EDGE_FEATURES:
        if ops is not None and 'edge_dir_0' in ops and ('edge_length_0' in ops):
            edge_dir = ops['edge_dir_0']
            edge_length = ops['edge_length_0']
        else:
            try:
                (edge_dir, edge_length) = build_edge_geometry(folder_path, name_prefix, folder_num, data_id, num_nodes)
            except (FileNotFoundError, ValueError, IndexError) as e:
                if is_main_process():
                    print(f'Warning: geometry edge features unavailable for {folder_num}-{data_id}: {e}')
    eps_data = load_file_data(folder_path, 'eps', folder_num, data_id, name_prefix)
    _require_rows('eps', eps_data, num_nodes)
    _require_feature_dim('eps', eps_data, 2)
    _require_rows('Eb', Eb_data, num_nodes)
    _require_feature_dim('Eb', Eb_data, 2)
    if b_vec.shape[0] != num_nodes:
        raise ValueError(f'b length mismatch: got {b_vec.shape[0]}, expected {num_nodes}')
    r_vec = b_vec - A_mat.dot(Eb_vec)
    current_field_data = torch.from_numpy(Eb_data)
    background_field_data = torch.from_numpy(Eb_data)
    x_tensor = torch.cat([torch.from_numpy(eps_data), torch.tensor(r_vec.real, dtype=torch.float32).unsqueeze(1), torch.tensor(r_vec.imag, dtype=torch.float32).unsqueeze(1), current_field_data, background_field_data], dim=1)
    if x_tensor.shape[1] != 8:
        raise ValueError(f'x feature dim mismatch: got {x_tensor.shape[1]}, expected 8')
    raw_y_data = load_file_data(folder_path, 'x', folder_num, data_id, name_prefix)
    _require_rows('x', raw_y_data, num_nodes)
    _require_feature_dim('x', raw_y_data, 2)
    y_tensor = torch.from_numpy(raw_y_data[:num_nodes])
    data = AMGData(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
    data.num_nodes_level_0 = int(x_tensor.shape[0])
    data.k_idx = torch.tensor([k])
    if NETWORK_USE_FREQUENCY_MODULATION:
        data.freq = _load_sample_frequency(folder_path, name_prefix, folder_num, data_id)
    data.A_train = A_train_tensor
    data.b_train = b_train_tensor
    data.b_train_scale = torch.tensor([b_train_scale], dtype=torch.float32)
    data.edge_attr_source = edge_attr_source
    if edge_dir is not None and edge_length is not None:
        data.edge_dir = edge_dir
        data.edge_length = edge_length
    if recovery_op is not None:
        data.recovery_operator = recovery_op
        data.recovery_type = recovery_type
    if ops is not None:
        max_levels = len([key for key in ops.keys() if key.startswith('R_')])
        data.num_amg_levels = max_levels
        for l in range(max_levels):
            R_l = ops[f'R_{l}']
            P_l = ops[f'P_{l}']
            data[f'R_{l}'] = R_l
            data[f'P_{l}'] = P_l
            data[f'num_nodes_level_{l}'] = int(R_l.shape[1])
            data[f'num_nodes_level_{l + 1}'] = int(R_l.shape[0])
            edge_index_next = ops[f'edge_index_{l + 1}']
            edge_attr_next = ops[f'edge_attr_{l + 1}']
            if edge_index_next.numel() > 0 and int(edge_index_next.max()) >= int(R_l.shape[0]):
                raise ValueError(f'AMG edge_index_{l + 1} references node outside coarse level: max_index={int(edge_index_next.max())}, nodes={int(R_l.shape[0])}')
            if edge_attr_next.shape[0] != edge_index_next.shape[1]:
                raise ValueError(f'AMG edge_attr_{l + 1} row count mismatch: edge_attr={tuple(edge_attr_next.shape)}, edge_index={tuple(edge_index_next.shape)}')
            data[f'edge_index_{l + 1}'] = edge_index_next
            data[f'edge_attr_{l + 1}'] = edge_attr_next
            if P_l.shape[0] != R_l.shape[1] or P_l.shape[1] != R_l.shape[0]:
                raise ValueError(f'AMG P/R shape mismatch at level {l}: R={tuple(R_l.shape)}, P={tuple(P_l.shape)}')
            validate_edge_attr_dim(data[f'edge_attr_{l + 1}'], f'edge_attr_{l + 1}')
            if NETWORK_USE_GEOMETRIC_EDGE_FEATURES and f'edge_dir_{l + 1}' in ops:
                if ops[f'edge_dir_{l + 1}'].shape[0] != int(R_l.shape[0]):
                    raise ValueError(f"AMG edge_dir_{l + 1} row count mismatch: edge_dir={tuple(ops[f'edge_dir_{l + 1}'].shape)}, nodes={int(R_l.shape[0])}")
                data[f'edge_dir_{l + 1}'] = ops[f'edge_dir_{l + 1}']
            if NETWORK_USE_GEOMETRIC_EDGE_FEATURES and f'edge_length_{l + 1}' in ops:
                if ops[f'edge_length_{l + 1}'].shape[0] != int(R_l.shape[0]):
                    raise ValueError(f"AMG edge_length_{l + 1} row count mismatch: edge_length={tuple(ops[f'edge_length_{l + 1}'].shape)}, nodes={int(R_l.shape[0])}")
                data[f'edge_length_{l + 1}'] = ops[f'edge_length_{l + 1}']
    else:
        if is_main_process():
            print(f'Warning: AMG operators file not found: {operator_file}')
        data.num_amg_levels = 0
    return data

def scan_all_data(root_path):
    global data_mapping, n_total
    if data_mapping and n_total > 0:
        return (data_mapping, n_total)
    data_mapping = {}
    k_idx = 1
    dataset_paths = _iter_dataset_paths(root_path)
    if is_main_process():
        print(f'Found {len(dataset_paths)} dataset paths for prefix {sca_prefix}: {[os.path.basename(p) or p for p in dataset_paths]}')
    total_folders = 0
    for dataset_path in dataset_paths:
        for (_, folder_num, folder_name, folder_path, name_prefix) in _collect_sample_folders(dataset_path):
            file_ids = _sample_file_ids(folder_path, name_prefix, folder_num)
            for data_id in file_ids:
                data_mapping[k_idx] = (folder_path, name_prefix, folder_num, str(data_id))
                k_idx += 1
            total_folders += 1
            if is_main_process():
                print(f'  {folder_name}: prefix={name_prefix}{folder_num}, samples={len(file_ids)}')
    n_total = k_idx - 1
    if is_main_process():
        print(f'Scan complete: {total_folders} folders, {n_total} samples.')
    return (data_mapping, n_total)

# Allow graph caches saved under the original module name to load locally.
import sys
sys.modules.setdefault('build_graph', sys.modules[__name__])
