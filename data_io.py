import os
import re
import numpy as np

def load_txt_2d(filepath, dtype=np.float32):
    data = np.loadtxt(filepath, dtype=dtype)
    return data.reshape(1, -1) if data.ndim == 1 else data
_PREFIX_ALIASES = {'vertex': ('vertex', 'vetex'), 'vetex': ('vertex', 'vetex')}

def prefixed_filename(prefix, sca_prefix, folder_num, data_id):
    return f'{prefix}_{sca_prefix}{folder_num}_{data_id}.txt'

def prefix_candidates(prefix):
    return _PREFIX_ALIASES.get(str(prefix).lower(), (prefix,))

def find_prefixed_file(folder_path, prefix, folder_num, data_id, sca_prefix):
    expected_names = [prefixed_filename(candidate, sca_prefix, folder_num, data_id) for candidate in prefix_candidates(prefix)]
    for expected_name in expected_names:
        filepath = os.path.join(folder_path, expected_name)
        if os.path.exists(filepath):
            return filepath
    expected_lower = {name.lower() for name in expected_names}
    for item in os.listdir(folder_path):
        if item.lower() in expected_lower:
            return os.path.join(folder_path, item)
    return None

def load_prefixed_txt(folder_path, prefix, folder_num, data_id, sca_prefix, dtype=np.float32):
    filepath = find_prefixed_file(folder_path, prefix, folder_num, data_id, sca_prefix)
    if filepath is None:
        expected = ', '.join((prefixed_filename(candidate, sca_prefix, folder_num, data_id) for candidate in prefix_candidates(prefix)))
        raise FileNotFoundError(f'File missing: expected one of [{expected}] in {folder_path}')
    return load_txt_2d(filepath, dtype=dtype)

def unpack_sample_info(info, default_prefix):
    if len(info) == 3:
        (folder_path, folder_num, data_id) = info
        return (folder_path, default_prefix, folder_num, data_id)
    if len(info) == 4:
        return info
    raise ValueError(f'Unsupported sample info format: {info}')

def sample_folder_specs(folder_name, sca_prefix):
    specs = []
    if sca_prefix.upper() == 'FSS':
        old_match = re.fullmatch('FSS_case(\\d+)_txt', folder_name, flags=re.IGNORECASE)
        if old_match:
            specs.append(('FSS', int(old_match.group(1)), 0))
            return specs
        new_match = re.fullmatch('FSS3x3_normal_case(\\d+)_txt', folder_name, flags=re.IGNORECASE)
        if new_match:
            specs.append(('A3N_FSS', int(new_match.group(1)), 0))
            return specs
        oblique_match = re.fullmatch('FSS3x3_oblique_case(\\d+)_txt', folder_name, flags=re.IGNORECASE)
        if oblique_match:
            specs.append(('A3O_FSS', int(oblique_match.group(1)), 0))
            return specs
        return specs
    match = re.fullmatch(f'{re.escape(sca_prefix)}(\\d+)_data', folder_name)
    if match:
        specs.append((sca_prefix, int(match.group(1)), 0))
    return specs

def sample_file_ids(folder_path, name_prefix, folder_num):
    pattern = re.compile(f'Aij_{re.escape(name_prefix)}{folder_num}_(\\d+)\\.txt', re.IGNORECASE)
    file_ids = []
    for item in os.listdir(folder_path):
        match = pattern.fullmatch(item)
        if match:
            file_ids.append(int(match.group(1)))
    return sorted(file_ids)
