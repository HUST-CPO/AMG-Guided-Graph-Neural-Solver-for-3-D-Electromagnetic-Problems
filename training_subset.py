import hashlib
import json
import math
import random

def mapping_digest(mapping):
    rows = [(int(k), list(mapping[k])) for k in sorted(mapping)]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=True).encode()).hexdigest()

def make_subset_manifest(train_k, test_k, fraction, seed, mapping=None):
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Training fraction must be in (0, 1].')
    if len(set(train_k)) != len(train_k) or len(set(test_k)) != len(test_k):
        raise ValueError('Duplicate indices in original split.')
    if set(train_k) & set(test_k):
        raise ValueError('Original train and test splits overlap.')
    count = int(len(train_k) * fraction)
    if count < 1:
        raise ValueError('Training subset would be empty.')
    selected = set(random.Random(seed).sample(list(train_k), count))
    result = {'version': 1, 'fraction_of_training_pool': fraction, 'subset_seed': seed, 'train_pool_k_indices': list(train_k), 'test_k_indices': list(test_k), 'train_k_indices': [k for k in train_k if k in selected], 'num_train_pool': len(train_k), 'num_train': count, 'num_test': len(test_k)}
    if mapping is not None:
        result['data_mapping_sha256'] = mapping_digest(mapping)
    return result

def make_dataset_subset_manifest(eligible_k, fraction, subset_seed, test_fraction=0.2, split_seed=42, mapping=None):
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Dataset fraction must be in (0, 1].')
    if not math.isfinite(test_fraction) or not 0 < test_fraction < 1:
        raise ValueError('Test fraction must be in (0, 1).')
    eligible_k = list(eligible_k)
    if len(set(eligible_k)) != len(eligible_k):
        raise ValueError('Duplicate indices in eligible pool.')
    subset_count = int(len(eligible_k) * fraction)
    if subset_count < 2:
        raise ValueError('Dataset subset is too small to split.')
    selected = set(random.Random(subset_seed).sample(eligible_k, subset_count))
    subset_k = [k for k in eligible_k if k in selected]
    test_count = math.ceil(subset_count * test_fraction)
    if test_count <= 0 or test_count >= subset_count:
        raise ValueError('Dataset subset cannot produce non-empty train and test splits.')
    test_selected = set(random.Random(split_seed).sample(subset_k, test_count))
    train_k = [k for k in subset_k if k not in test_selected]
    test_k = [k for k in subset_k if k in test_selected]
    result = {'version': 2, 'sampling_protocol': 'sample_eligible_pool_then_split', 'fraction_of_eligible_pool': fraction, 'subset_seed': subset_seed, 'test_fraction_of_subset': test_fraction, 'split_seed': split_seed, 'eligible_pool_k_indices': eligible_k, 'subset_k_indices': subset_k, 'train_k_indices': train_k, 'test_k_indices': test_k, 'num_eligible_pool': len(eligible_k), 'num_subset': subset_count, 'num_train': len(train_k), 'num_test': len(test_k)}
    if mapping is not None:
        result['data_mapping_sha256'] = mapping_digest(mapping)
    return result

def load_training_subset(train_k, test_k, filename, mapping=None):
    if not filename:
        return (train_k, test_k, None)
    with open(filename, encoding='utf-8') as stream:
        result = json.load(stream)
    version = result.get('version')
    if version == 1:
        if result['train_pool_k_indices'] != list(train_k):
            raise ValueError('Training pool differs from the subset manifest.')
        if result['test_k_indices'] != list(test_k):
            raise ValueError('Test split differs from the subset manifest.')
        expected = make_subset_manifest(train_k, test_k, result['fraction_of_training_pool'], result['subset_seed'], mapping)
        for (key, value) in expected.items():
            if result.get(key) != value:
                raise ValueError(f'Training subset manifest mismatch: {key}')
        return (result['train_k_indices'], list(test_k), result)
    if version == 2:
        eligible_k = result['eligible_pool_k_indices']
        runtime_eligible = list(train_k) + list(test_k)
        if len(runtime_eligible) != len(eligible_k) or set(runtime_eligible) != set(eligible_k):
            raise ValueError('Eligible data pool differs from the dataset subset manifest.')
        expected = make_dataset_subset_manifest(eligible_k, result['fraction_of_eligible_pool'], result['subset_seed'], result['test_fraction_of_subset'], result['split_seed'], mapping)
        for (key, value) in expected.items():
            if result.get(key) != value:
                raise ValueError(f'Dataset subset manifest mismatch: {key}')
        return (result['train_k_indices'], result['test_k_indices'], result)
    raise ValueError('Unsupported subset manifest version.')
