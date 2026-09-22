from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.paths import project_path
import gc
import csv
import json
import os
import time
import numpy as np
import scipy.sparse as sp
import torch
from pyamg.aggregation.aggregation import standard_aggregation
from pyamg.strength import symmetric_strength_of_connection
from training.config import DATA_ROOT_PATH, SCA_PREFIX, EDGE_FEATURE_MATRIX, PRECOND_METHOD
from training.data_io import load_txt_2d, unpack_sample_info
from training.build_graph import scan_all_data
root_data_path = str(project_path(DATA_ROOT_PATH))
sca_prefix = SCA_PREFIX
THETA = 0.24
NUM_LEVELS = 3
PRECOND_METHOD_FOR_AMG = PRECOND_METHOD
AMG_METHOD_LABEL = PRECOND_METHOD_FOR_AMG if PRECOND_METHOD_FOR_AMG else 'raw'
SAVE_AS_LEGACY_AMG_FILENAME = os.environ.get('FEMGRL_AMG_SAVE_LEGACY_FILENAME', '1').strip().lower() in {'1', 'true', 'yes', 'y', 'on'}
AMG_SKIP_EXISTING = os.environ.get('FEMGRL_AMG_SKIP_EXISTING', os.environ.get('FEMGRL_PREPROCESS_SKIP_EXISTING', '0')).strip().lower() in {'1', 'true', 'yes', 'y', 'on'}

def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for (k, v) in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

def _abs_distribution(values):
    values = np.asarray(values)
    if values.size == 0:
        return {'abs_min': 0.0, 'abs_max': 0.0, 'abs_mean': 0.0, 'abs_std': 0.0, 'abs_p50': 0.0, 'abs_p95': 0.0, 'abs_l2': 0.0, 'zero_count': 0}
    abs_values = np.abs(values.reshape(-1))
    return {'abs_min': float(abs_values.min()), 'abs_max': float(abs_values.max()), 'abs_mean': float(abs_values.mean()), 'abs_std': float(abs_values.std()), 'abs_p50': float(np.percentile(abs_values, 50)), 'abs_p95': float(np.percentile(abs_values, 95)), 'abs_l2': float(np.linalg.norm(abs_values)), 'zero_count': int(np.count_nonzero(abs_values == 0))}

def sparse_matrix_stats_dict(A_csr):
    A_csr = A_csr.tocsr()
    (n_rows, n_cols) = A_csr.shape
    stats = {'rows': int(n_rows), 'cols': int(n_cols), 'nnz': int(A_csr.nnz), 'density': float(A_csr.nnz / max(n_rows * n_cols, 1))}
    stats.update(_abs_distribution(A_csr.data))
    diag_abs = np.abs(A_csr.diagonal())
    if diag_abs.size > 0:
        stats.update({'diag_abs_min': float(diag_abs.min()), 'diag_abs_max': float(diag_abs.max()), 'diag_abs_mean': float(diag_abs.mean()), 'diag_abs_p05': float(np.percentile(diag_abs, 5)), 'diag_abs_p95': float(np.percentile(diag_abs, 95)), 'near_zero_diag_count': int(np.count_nonzero(diag_abs < 1e-12))})
    return stats

def vector_stats_dict(vec):
    if vec is None:
        return {'length': 0, 'present': False}
    vec = np.asarray(vec).reshape(-1)
    stats = {'length': int(vec.size), 'present': True}
    stats.update(_abs_distribution(vec))
    return stats

def add_prefixed_stats(row, prefix, stats):
    for (key, value) in stats.items():
        row[f'{prefix}_{key}'] = value

def write_report_files(report_dir, base_name, rows, summary):
    os.makedirs(report_dir, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(report_dir, f'{base_name}_{timestamp}.csv')
    json_path = os.path.join(report_dir, f'{base_name}_{timestamp}.json')
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    payload = {'summary': _to_jsonable(summary), 'records': _to_jsonable(rows)}
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return (csv_path, json_path)

def _torch_sparse_to_scipy_csr(torch_mat):
    if not isinstance(torch_mat, torch.Tensor):
        raise TypeError(f'A_new 必须是 torch.Tensor，实际类型: {type(torch_mat)}')
    torch_mat = torch_mat.detach().cpu()
    is_sparse_layout = getattr(torch_mat, 'is_sparse', False) or getattr(torch_mat, 'is_sparse_csr', False) or str(getattr(torch_mat, 'layout', '')).startswith('torch.sparse_')
    if is_sparse_layout:
        if not getattr(torch_mat, 'is_sparse', False):
            torch_mat = torch_mat.to_sparse_coo()
        torch_mat = torch_mat.coalesce()
        indices = torch_mat.indices().numpy()
        values = torch_mat.values().numpy()
        shape = tuple(torch_mat.shape)
        return sp.coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr()
    dense = torch_mat.numpy()
    return sp.csr_matrix(dense)

def _load_preconditioned_A(folder_path, folder_num, data_id, name_prefix=None):
    if not PRECOND_METHOD_FOR_AMG:
        raise ValueError('PRECOND_METHOD_FOR_AMG 为空，无法定位 precond_*.pt 文件。')
    precond_file = os.path.join(folder_path, f'precond_{PRECOND_METHOD_FOR_AMG}_{name_prefix or sca_prefix}{folder_num}_{data_id}.pt')
    if not os.path.exists(precond_file):
        raise FileNotFoundError(f'预处理矩阵文件缺失: {precond_file}')
    try:
        pt_data = torch.load(precond_file, map_location='cpu', weights_only=True)
    except TypeError:
        pt_data = torch.load(precond_file, map_location='cpu')
    if 'A_new' not in pt_data:
        raise KeyError(f'预处理文件中缺少 A_new: {precond_file}')
    A_new = _torch_sparse_to_scipy_csr(pt_data['A_new'])
    if A_new.shape[0] != A_new.shape[1]:
        raise ValueError(f'A_new 必须是方阵，实际形状: {A_new.shape}')
    return A_new

def _load_case_insensitive_prefixed_txt(folder_path, prefix_candidates, folder_num, data_id, name_prefix, dtype=np.float32):
    expected_names = [f'{prefix}_{name_prefix}{folder_num}_{data_id}.txt' for prefix in prefix_candidates]
    for expected_name in expected_names:
        direct_path = os.path.join(folder_path, expected_name)
        if os.path.exists(direct_path):
            return load_txt_2d(direct_path, dtype=dtype)
    expected_lookup = {name.lower(): name for name in expected_names}
    for item in os.listdir(folder_path):
        if item.lower() in expected_lookup:
            return load_txt_2d(os.path.join(folder_path, item), dtype=dtype)
    expected_display = ', '.join(expected_names)
    raise FileNotFoundError(f'File missing: expected one of [{expected_display}] in {folder_path}')

def _load_edge_txt(folder_path, folder_num, data_id, name_prefix, dtype=np.float32):
    return _load_case_insensitive_prefixed_txt(folder_path, ('edge',), folder_num, data_id, name_prefix, dtype=dtype)

def _load_vertex_txt(folder_path, folder_num, data_id, name_prefix, dtype=np.float32):
    return _load_case_insensitive_prefixed_txt(folder_path, ('vertex', 'vetex'), folder_num, data_id, name_prefix, dtype=dtype)

def _load_raw_A(folder_path, folder_num, data_id, name_prefix):
    Aij_data = _load_case_insensitive_prefixed_txt(folder_path, ('Aij',), folder_num, data_id, name_prefix, dtype=np.float64)
    Av_data = _load_case_insensitive_prefixed_txt(folder_path, ('Av',), folder_num, data_id, name_prefix, dtype=np.float64)
    if Aij_data.shape[0] != Av_data.shape[0]:
        raise ValueError(f'Aij/Av row count mismatch: Aij={Aij_data.shape[0]}, Av={Av_data.shape[0]}')
    rows = Aij_data[:, 0].astype(int) - 1
    cols = Aij_data[:, 1].astype(int) - 1
    num_nodes = max(rows.max(), cols.max()) + 1
    values = Av_data[:, 0] + 1j * Av_data[:, 1]
    return sp.coo_matrix((values, (rows, cols)), shape=(num_nodes, num_nodes)).tocsr()

def _load_amg_base_A(folder_path, folder_num, data_id, name_prefix):
    if PRECOND_METHOD_FOR_AMG:
        return _load_preconditioned_A(folder_path, folder_num, data_id, name_prefix)
    return _load_raw_A(folder_path, folder_num, data_id, name_prefix)

def scipy_sparse_to_torch(scipy_mat):
    scipy_mat = scipy_mat.tocoo()
    indices = torch.vstack((torch.tensor(scipy_mat.row, dtype=torch.long), torch.tensor(scipy_mat.col, dtype=torch.long)))
    values = torch.tensor(scipy_mat.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values, torch.Size(scipy_mat.shape)).coalesce()

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

def build_graph_from_sparse_matrix(A_mat):
    A_csr = A_mat.tocsr()
    coo = A_csr.tocoo()
    rows = coo.row.astype(np.int64)
    cols = coo.col.astype(np.int64)
    values = coo.data
    mask = rows != cols
    rows = rows[mask]
    cols = cols[mask]
    values = values[mask]
    diag = A_csr.diagonal().astype(np.complex128)
    edge_values = _select_edge_complex_values(rows, values, diag)
    edge_index = torch.tensor(np.vstack([cols, rows]), dtype=torch.long)
    edge_attr = torch.tensor(_encode_mag_phase(edge_values), dtype=torch.float32)
    return (edge_index, edge_attr)

def aggregate_edge_geometry(edge_map, current_edge_vec, current_edge_length):
    num_coarse_edges = int(edge_map.max()) + 1
    coarse_edge_vec = np.zeros((num_coarse_edges, current_edge_vec.shape[1]), dtype=np.float32)
    coarse_edge_length = np.zeros((num_coarse_edges, 1), dtype=np.float32)
    for cid in range(num_coarse_edges):
        members = np.where(edge_map == cid)[0]
        if members.size == 0:
            continue
        member_vecs = current_edge_vec[members].copy()
        member_len = current_edge_length[members].reshape(-1, 1)
        ref = member_vecs[0].copy()
        ref_norm = np.linalg.norm(ref)
        if ref_norm < 1e-12:
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            ref_norm = 1.0
        ref = ref / ref_norm
        for k in range(member_vecs.shape[0]):
            if np.dot(member_vecs[k], ref) < 0:
                member_vecs[k] *= -1.0
        weighted_sum = (member_vecs * np.maximum(member_len, 1e-12)).sum(axis=0)
        norm_sum = np.linalg.norm(weighted_sum)
        if norm_sum < 1e-12:
            mean_vec = member_vecs.mean(axis=0)
            mean_norm = np.linalg.norm(mean_vec)
            if mean_norm < 1e-12:
                mean_vec = ref
                mean_norm = 1.0
            coarse_dir = mean_vec / mean_norm
            coarse_len = float(np.mean(member_len))
        else:
            coarse_dir = weighted_sum / norm_sum
            coarse_len = norm_sum / max(float(member_len.sum()), 1e-12) * float(np.mean(member_len))
        coarse_edge_vec[cid] = coarse_dir * coarse_len
        coarse_edge_length[cid, 0] = max(coarse_len, 1e-12)
    coarse_edge_dir = coarse_edge_vec / (coarse_edge_length + 1e-12)
    return (coarse_edge_vec, coarse_edge_dir, coarse_edge_length)

def precompute_edge_space_operators(folder_path, folder_num, data_id, verbose=False, name_prefix=None):
    name_prefix = name_prefix or sca_prefix
    sample_name = f'{name_prefix}{folder_num}_{data_id}'
    report = {'sample': sample_name, 'name_prefix': name_prefix, 'folder_num': folder_num, 'data_id': data_id, 'folder_path': folder_path, 'method': AMG_METHOD_LABEL, 'theta': THETA, 'num_levels': NUM_LEVELS, 'status': 'started'}
    start_time = time.perf_counter()
    if verbose:
        print(f'\n---> [边空间直接聚合 | 矩阵 {AMG_METHOD_LABEL}] 开始处理: 文件夹 {folder_num}, 数据 ID {data_id}')
    try:
        edge_data = _load_edge_txt(folder_path, folder_num, data_id, name_prefix, dtype=np.float32)
        vertex_data = _load_vertex_txt(folder_path, folder_num, data_id, name_prefix, dtype=np.float32)
        A_edge_phys = _load_amg_base_A(folder_path, folder_num, data_id, name_prefix)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as e:
        if verbose:
            print(f'  [跳过] {e}')
        report['status'] = 'skipped'
        report['error'] = str(e)
        report['elapsed_sec'] = time.perf_counter() - start_time
        return report
    num_edges = A_edge_phys.shape[0]
    report['status'] = 'loaded'
    report['edge_file_rows'] = int(edge_data.shape[0])
    report['vertex_file_rows'] = int(vertex_data.shape[0])
    report['level0_dof'] = int(num_edges)
    add_prefixed_stats(report, 'level0_A', sparse_matrix_stats_dict(A_edge_phys))
    A_edge_topo = A_edge_phys.copy()
    A_edge_topo.data = np.abs(A_edge_topo.data)
    v1_idx = edge_data[:, 0].astype(int) - 1
    v2_idx = edge_data[:, 1].astype(int) - 1
    coord1 = vertex_data[v1_idx]
    coord2 = vertex_data[v2_idx]
    current_edge_vec = coord2 - coord1
    current_edge_length = np.linalg.norm(current_edge_vec, axis=1, keepdims=True)
    add_prefixed_stats(report, 'level0_edge_length', vector_stats_dict(current_edge_length))
    if current_edge_vec.shape[0] != num_edges:
        raise ValueError(f'几何边数量与 A_new 维度不一致: edge={current_edge_vec.shape[0]}, A_new={num_edges}')
    operators = {}
    (edge_index_0, edge_attr_0) = build_graph_from_sparse_matrix(A_edge_phys)
    report['level0_graph_edges'] = int(edge_index_0.shape[1])
    report['level0_edge_attr_dim'] = int(edge_attr_0.shape[1]) if edge_attr_0.ndim > 1 else 0
    report['level0_geometry_edges'] = int(current_edge_vec.shape[0])
    operators['edge_index_0'] = edge_index_0
    operators['edge_attr_0'] = edge_attr_0
    operators['edge_dir_0'] = torch.tensor(current_edge_vec / (current_edge_length + 1e-12), dtype=torch.float32)
    operators['edge_length_0'] = torch.tensor(current_edge_length, dtype=torch.float32)
    for l in range(NUM_LEVELS - 1):
        operators[f'A_real_{l}'] = scipy_sparse_to_torch(A_edge_phys.real)
        operators[f'A_imag_{l}'] = scipy_sparse_to_torch(A_edge_phys.imag)
        C_matrix = symmetric_strength_of_connection(A_edge_topo, theta=THETA)
        report[f'level{l}_fine_dof'] = int(num_edges)
        report[f'level{l}_strength_nnz'] = int(C_matrix.nnz)
        Agg_out = standard_aggregation(C_matrix)
        Agg_e = Agg_out[0] if isinstance(Agg_out, tuple) else Agg_out
        num_coarse_edges = Agg_e.shape[1]
        report[f'level{l}_coarse_dof'] = int(num_coarse_edges)
        report[f'level{l}_coarsening_ratio'] = float(num_coarse_edges / max(num_edges, 1))
        edge_map = np.full(num_edges, -1, dtype=np.int64)
        (rows_agg, cols_agg) = Agg_e.nonzero()
        edge_map[rows_agg] = cols_agg
        if num_coarse_edges == 0:
            raise RuntimeError(f'Level {l} 粗边数量为 0，无法继续粗化。')
        valid_mask = edge_map >= 0
        report[f'level{l}_aggregate_members'] = int(valid_mask.sum())
        report[f'level{l}_unaggregated_edges'] = int(num_edges - valid_mask.sum())
        R_rows = edge_map[valid_mask]
        R_cols = np.arange(num_edges, dtype=np.int64)[valid_mask]
        R_vals = np.ones(valid_mask.sum(), dtype=np.float32)
        R_edge_scipy = sp.coo_matrix((R_vals, (R_rows, R_cols)), shape=(num_coarse_edges, num_edges)).tocsr()
        P_edge_scipy = R_edge_scipy.transpose().tocsr()
        operators[f'R_{l}'] = scipy_sparse_to_torch(R_edge_scipy)
        operators[f'P_{l}'] = scipy_sparse_to_torch(P_edge_scipy)
        A_edge_phys_next = R_edge_scipy.dot(A_edge_phys.dot(P_edge_scipy)).tocsr()
        add_prefixed_stats(report, f'level{l + 1}_A', sparse_matrix_stats_dict(A_edge_phys_next))
        A_edge_topo_next = A_edge_phys_next.copy()
        A_edge_topo_next.data = np.abs(A_edge_topo_next.data)
        (coarse_edge_vec, coarse_edge_dir, coarse_edge_length) = aggregate_edge_geometry(edge_map, current_edge_vec, current_edge_length)
        (edge_index_next, edge_attr_next) = build_graph_from_sparse_matrix(A_edge_phys_next)
        report[f'level{l + 1}_dof'] = int(num_coarse_edges)
        report[f'level{l + 1}_graph_edges'] = int(edge_index_next.shape[1])
        report[f'level{l + 1}_edge_attr_dim'] = int(edge_attr_next.shape[1]) if edge_attr_next.ndim > 1 else 0
        report[f'level{l + 1}_geometry_edges'] = int(coarse_edge_dir.shape[0])
        operators[f'edge_index_{l + 1}'] = edge_index_next
        operators[f'edge_attr_{l + 1}'] = edge_attr_next
        add_prefixed_stats(report, f'level{l + 1}_edge_length', vector_stats_dict(coarse_edge_length))
        operators[f'edge_dir_{l + 1}'] = torch.tensor(coarse_edge_dir, dtype=torch.float32)
        operators[f'edge_length_{l + 1}'] = torch.tensor(coarse_edge_length, dtype=torch.float32)
        operators[f'A_real_{l + 1}'] = scipy_sparse_to_torch(A_edge_phys_next.real)
        operators[f'A_imag_{l + 1}'] = scipy_sparse_to_torch(A_edge_phys_next.imag)
        if verbose:
            print(f'  [Level {l}] 边自由度: {num_edges} -> {num_coarse_edges}')
            print(f'  [Level {l + 1}] coarse graph 边数: {edge_index_next.shape[1]} | edge_attr维度: {edge_attr_next.shape[1]}')
            print(f'  [Level {l + 1}] coarse geometry 节点数: {coarse_edge_dir.shape[0]}')
        if l < NUM_LEVELS - 2:
            A_edge_phys = A_edge_phys_next
            A_edge_topo = A_edge_topo_next
            current_edge_vec = coarse_edge_vec
            current_edge_length = coarse_edge_length
            num_edges = num_coarse_edges
    if SAVE_AS_LEGACY_AMG_FILENAME:
        save_name = f'amg_operators_{name_prefix}{folder_num}_{data_id}.pt'
    else:
        save_name = f'amg_operators_{AMG_METHOD_LABEL}_{name_prefix}{folder_num}_{data_id}.pt'
    save_path = os.path.join(folder_path, save_name)
    torch.save(operators, save_path)
    report['status'] = 'success'
    report['save_path'] = save_path
    report['elapsed_sec'] = time.perf_counter() - start_time
    if verbose:
        print(f'  [成功] 基于 {AMG_METHOD_LABEL} 矩阵的边空间粗化算子已保存至: {os.path.basename(save_path)}\n')
    gc.collect()
    return report

def run_offline_preprocessing(root_path):
    print(f'========== Start 3D edge-space pyamg offline preprocessing | matrix={AMG_METHOD_LABEL} ==========')
    (data_mapping, n_total) = scan_all_data(root_path)
    global_count = 0
    report_rows = []
    report_dir = os.path.join(root_path, 'preprocess_reports')
    for k_idx in range(1, n_total + 1):
        if k_idx not in data_mapping:
            continue
        (folder_path, name_prefix, folder_num, data_id) = unpack_sample_info(data_mapping[k_idx], sca_prefix)
        global_count += 1
        if global_count % 30 == 0:
            print(f'[Progress] processed {global_count} graphs.')
        is_verbose = global_count % 30 == 1
        save_name = f'amg_operators_{name_prefix}{folder_num}_{data_id}.pt' if SAVE_AS_LEGACY_AMG_FILENAME else f'amg_operators_{AMG_METHOD_LABEL}_{name_prefix}{folder_num}_{data_id}.pt'
        save_path = os.path.join(folder_path, save_name)
        if AMG_SKIP_EXISTING and os.path.exists(save_path):
            report_rows.append({'sample': f'{name_prefix}{folder_num}_{data_id}', 'name_prefix': name_prefix, 'folder_num': folder_num, 'data_id': data_id, 'folder_path': folder_path, 'method': AMG_METHOD_LABEL, 'theta': THETA, 'num_levels': NUM_LEVELS, 'status': 'skipped_existing', 'save_path': save_path, 'sample_index': global_count, 'k_idx': k_idx})
            if is_verbose:
                print(f'[Skip] existing AMG operators: {save_path}')
            continue
        try:
            report = precompute_edge_space_operators(folder_path, folder_num, str(data_id), verbose=is_verbose, name_prefix=name_prefix)
        except Exception as e:
            print(f'\n[Error] failed on {name_prefix}{folder_num}-{data_id}: {e}')
            report = {'sample': f'{name_prefix}{folder_num}_{data_id}', 'name_prefix': name_prefix, 'folder_num': folder_num, 'data_id': data_id, 'folder_path': folder_path, 'method': AMG_METHOD_LABEL, 'theta': THETA, 'num_levels': NUM_LEVELS, 'status': 'error', 'error': str(e)}
        if not isinstance(report, dict):
            report = {'sample': f'{name_prefix}{folder_num}_{data_id}', 'name_prefix': name_prefix, 'folder_num': folder_num, 'data_id': data_id, 'folder_path': folder_path, 'method': AMG_METHOD_LABEL, 'theta': THETA, 'num_levels': NUM_LEVELS, 'status': 'success' if report else 'skipped'}
        report['sample_index'] = global_count
        report['k_idx'] = k_idx
        report_rows.append(report)
    status_counts = {}
    for row in report_rows:
        status = row.get('status', 'unknown')
        status_counts[status] = status_counts.get(status, 0) + 1
    ratio_summary = {}
    for level in range(NUM_LEVELS - 1):
        key = f'level{level}_coarsening_ratio'
        values = [float(row[key]) for row in report_rows if row.get('status') == 'success' and key in row]
        if values:
            ratio_summary[f'{key}_mean'] = float(np.mean(values))
            ratio_summary[f'{key}_min'] = float(np.min(values))
            ratio_summary[f'{key}_max'] = float(np.max(values))
    report_summary = {'script': 'amg_preprocess.py', 'data_root': root_path, 'dataset_prefix': sca_prefix, 'total_scanned': n_total, 'total_records': len(report_rows), 'status_counts': status_counts, 'coarsening_ratio_summary': ratio_summary, 'parameters': {'PRECOND_METHOD_FOR_AMG': AMG_METHOD_LABEL, 'THETA': THETA, 'NUM_LEVELS': NUM_LEVELS, 'EDGE_FEATURE_MATRIX': EDGE_FEATURE_MATRIX, 'SAVE_AS_LEGACY_AMG_FILENAME': SAVE_AS_LEGACY_AMG_FILENAME}}
    (csv_path, json_path) = write_report_files(report_dir, f'amg_preprocess_eval_{AMG_METHOD_LABEL}_{sca_prefix}', report_rows, report_summary)
    print(f'[Report] AMG preprocess CSV saved to: {csv_path}')
    print(f'[Report] AMG preprocess JSON saved to: {json_path}')
    print(f'[Report] status counts: {status_counts}')
    if ratio_summary:
        print(f'[Report] coarsening ratio summary: {ratio_summary}')
if __name__ == '__main__':
    import argparse
    argparse.ArgumentParser(description='Generate DiagScale AMG operators').parse_args()
    run_offline_preprocessing(root_data_path)
    print('\n========== 🎉 直接边空间 pyamg 粗化算子全部生成完成！==========')
