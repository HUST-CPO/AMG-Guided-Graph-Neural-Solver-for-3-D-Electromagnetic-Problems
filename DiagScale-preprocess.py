from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import os
import numpy as np
import scipy.sparse as sp
import torch
from training.config import DATA_ROOT_PATH, SCA_PREFIX, DATASET_DIRS_PATTERN
from training.paths import project_path
from training.data_io import load_prefixed_txt, sample_file_ids as data_sample_file_ids, sample_folder_specs as data_sample_folder_specs, unpack_sample_info
DIAG_SCALE_FLOOR = 1e-12
TORCH_COMPLEX_DTYPE = torch.complex128
def _sample_folder_specs(folder_name):
    return data_sample_folder_specs(folder_name, SCA_PREFIX)

def _iter_dataset_paths(root_path):
    paths = []
    root_name = os.path.basename(os.path.normpath(root_path))
    if root_name and _sample_folder_specs(root_name):
        paths.append(root_path)
    if any((os.path.isdir(os.path.join(root_path, item)) and _sample_folder_specs(item) for item in os.listdir(root_path))):
        paths.append(root_path)
    paths.extend((os.path.join(root_path, item) for item in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, item)) and (not _sample_folder_specs(item)) and item.startswith(DATASET_DIRS_PATTERN.replace('*', ''))))
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

def scan_preprocess_data(root_path):
    data_mapping = {}
    k_idx = 1
    for dataset_path in _iter_dataset_paths(root_path):
        for (_, folder_num, _folder_name, folder_path, name_prefix) in _collect_sample_folders(dataset_path):
            for data_id in _sample_file_ids(folder_path, name_prefix, folder_num):
                data_mapping[k_idx] = (folder_path, name_prefix, folder_num, str(data_id))
                k_idx += 1
    return (data_mapping, k_idx - 1)

def scipy_csr_to_petsc(A_csr):
    A_petsc = PETSc.Mat().createAIJ(size=A_csr.shape, csr=(A_csr.indptr, A_csr.indices, A_csr.data))
    A_petsc.assemble()
    return A_petsc

def petsc_to_scipy_csr(A_petsc):
    (indptr, indices, data) = A_petsc.getValuesCSR()
    return sp.csr_matrix((data, indices, indptr), shape=A_petsc.getSize(), dtype=np.complex128)

def numpy_to_petsc_vec(b_np):
    b_np = np.asarray(b_np, dtype=np.complex128).flatten()
    vec = PETSc.Vec().createSeq(b_np.size)
    vec.setValues(range(b_np.size), b_np)
    vec.assemble()
    return vec

def parse_complex_vector(txt_data, n, default_value=0.0):
    if txt_data.shape[1] >= 2:
        vec = txt_data[:, 0] + 1j * txt_data[:, 1]
    else:
        vec = txt_data[:, 0].astype(np.complex128)
    vec = np.asarray(vec, dtype=np.complex128).flatten()
    if vec.size != n:
        raise ValueError(f'向量长度不匹配: expected {n}, got {vec.size}')
    if not np.all(np.isfinite(vec)):
        raise ValueError('向量中包含 NaN 或 Inf')
    return vec

def save_operator_to_pt(A_new, b_new, Eb_new, recovery_operator, folder_path, folder_num, data_id, method_name, name_prefix=None):
    operators = {}
    A_new = A_new.tocsr().astype(np.complex128)
    A_new_coo = A_new.tocoo()
    indices = torch.vstack([torch.tensor(A_new_coo.row, dtype=torch.long), torch.tensor(A_new_coo.col, dtype=torch.long)])
    values = torch.tensor(A_new_coo.data, dtype=TORCH_COMPLEX_DTYPE)
    operators['A_new'] = torch.sparse_coo_tensor(indices, values, torch.Size(A_new_coo.shape)).coalesce()
    operators['b_new'] = torch.tensor(np.asarray(b_new, dtype=np.complex128).flatten(), dtype=TORCH_COMPLEX_DTYPE)
    operators['Eb_new'] = torch.tensor(np.asarray(Eb_new, dtype=np.complex128).flatten(), dtype=TORCH_COMPLEX_DTYPE)
    if sp.issparse(recovery_operator):
        R_coo = recovery_operator.tocoo()
        r_indices = torch.vstack([torch.tensor(R_coo.row, dtype=torch.long), torch.tensor(R_coo.col, dtype=torch.long)])
        r_values = torch.tensor(R_coo.data, dtype=TORCH_COMPLEX_DTYPE)
        operators['recovery_operator'] = torch.sparse_coo_tensor(r_indices, r_values, torch.Size(R_coo.shape)).coalesce()
        operators['recovery_type'] = 'sparse_matrix'
    else:
        operators['recovery_operator'] = torch.tensor(np.asarray(recovery_operator, dtype=np.complex128).flatten(), dtype=TORCH_COMPLEX_DTYPE)
        operators['recovery_type'] = 'vector_diagonal'
    save_path = os.path.join(folder_path, f'precond_{method_name}_{name_prefix or SCA_PREFIX}{folder_num}_{data_id}.pt')
    torch.save(operators, save_path)

def diag_scaling_op(A_petsc, b_petsc, Eb_petsc):
    v_scale = A_petsc.createVecLeft()
    A_petsc.getDiagonal(v_scale)
    with v_scale as arr:
        diag_values = arr.copy()
        scale = 1.0 / np.sqrt(np.maximum(np.abs(diag_values), DIAG_SCALE_FLOOR))
        scale = np.asarray(scale, dtype=arr.dtype)
        if not np.all(np.isfinite(scale)):
            raise ValueError('DiagScale 产生了 NaN 或 Inf 缩放因子')
        arr[:] = scale
    A_new = A_petsc.duplicate(PETSc.Mat.DuplicateOption.COPY_VALUES)
    A_new.diagonalScale(L=v_scale, R=v_scale)
    b_new = b_petsc.duplicate()
    b_new.pointwiseMult(b_petsc, v_scale)
    recovery_op = v_scale.getArray().copy()
    x0 = Eb_petsc.getArray().copy()
    Eb_ret = x0 / recovery_op
    A_ret = petsc_to_scipy_csr(A_new)
    b_ret = b_new.getArray().copy()
    A_new.destroy()
    b_new.destroy()
    v_scale.destroy()
    return (A_ret, b_ret, Eb_ret, recovery_op)

def run_batch_preprocessing(skip_existing=True):
    root = str(project_path(DATA_ROOT_PATH))
    mapping, _ = scan_preprocess_data(root)
    written = skipped = 0
    for info in mapping.values():
        folder, prefix, folder_num, data_id = unpack_sample_info(info, SCA_PREFIX)
        output = os.path.join(folder, f'precond_DiagScale_{prefix}{folder_num}_{data_id}.pt')
        if skip_existing and os.path.exists(output):
            skipped += 1
            continue
        aij = load_prefixed_txt(folder, 'Aij', folder_num, data_id, prefix)
        av = load_prefixed_txt(folder, 'Av', folder_num, data_id, prefix)
        if aij.shape[1] < 2 or av.shape[1] < 2 or len(aij) != len(av) or len(aij) == 0:
            raise ValueError('Invalid Aij/Av matrix data')
        rows, cols = aij[:, 0].astype(int) - 1, aij[:, 1].astype(int) - 1
        n = int(max(rows.max(), cols.max()) + 1)
        values = (av[:, 0] + 1j * av[:, 1]).astype(np.complex128)
        matrix = sp.coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
        try:
            b = parse_complex_vector(load_prefixed_txt(folder, 'b', folder_num, data_id, prefix), n)
        except FileNotFoundError:
            b = np.ones(n, dtype=np.complex128)
        try:
            initial = parse_complex_vector(load_prefixed_txt(folder, 'Eb', folder_num, data_id, prefix), n)
        except FileNotFoundError:
            initial = np.zeros(n, dtype=np.complex128)
        objects = []
        try:
            a_petsc = scipy_csr_to_petsc(matrix)
            objects.append(a_petsc)
            b_petsc = numpy_to_petsc_vec(b)
            objects.append(b_petsc)
            initial_petsc = numpy_to_petsc_vec(initial)
            objects.append(initial_petsc)
            a_new, b_new, e_new, recovery = diag_scaling_op(a_petsc, b_petsc, initial_petsc)
            save_operator_to_pt(a_new, b_new, e_new, recovery, folder, folder_num,
                                data_id, 'DiagScale', name_prefix=prefix)
            written += 1
        finally:
            for obj in reversed(objects):
                obj.destroy()
    print(f'DiagScale: written={written}, skipped={skipped}')


def main():
    parser = argparse.ArgumentParser(description='Final PETSc DiagScale preprocessing')
    parser.add_argument('--overwrite', action='store_true', help='Rebuild existing DiagScale files')
    args, petsc_args = parser.parse_known_args()
    import petsc4py
    petsc4py.init(petsc_args)
    global PETSc
    from petsc4py import PETSc
    skip = os.environ.get('FEMGRL_PREPROCESS_SKIP_EXISTING', '1').lower() in {'1', 'true', 'yes', 'on'}
    run_batch_preprocessing(skip_existing=skip and not args.overwrite)


if __name__ == '__main__':
    main()
