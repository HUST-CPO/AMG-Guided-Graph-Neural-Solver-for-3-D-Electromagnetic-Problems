from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import argparse
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from training.build_graph import build_graph_data, get_graph_cache_dir, get_graph_cache_path, root_data_path, save_graph_data_cache, scan_all_data

def _data_stats(data):
    stats = {'num_nodes': int(data.x.size(0)) if hasattr(data, 'x') else 0, 'level0_edges': int(data.edge_index.size(1)) if hasattr(data, 'edge_index') else 0, 'num_amg_levels': int(getattr(data, 'num_amg_levels', 0))}
    A_train = getattr(data, 'A_train', None)
    if A_train is not None:
        A_train = A_train.coalesce() if getattr(A_train, 'is_sparse', False) else A_train
        stats['A_train_nnz'] = int(A_train._nnz()) if getattr(A_train, 'is_sparse', False) else int(A_train.numel())
    b_train = getattr(data, 'b_train', None)
    if b_train is not None:
        stats['b_train_size'] = int(b_train.numel())
    freq = getattr(data, 'freq', None)
    if freq is not None:
        stats['freq_hz'] = float(freq.reshape(-1)[0].item())
    for level in range(1, stats['num_amg_levels'] + 1):
        edge_index = getattr(data, f'edge_index_{level}', None)
        stats[f'level{level}_edges'] = int(edge_index.size(1)) if edge_index is not None else 0
        stats[f'level{level}_nodes'] = int(getattr(data, f'num_nodes_level_{level}', 0))
    return stats

def _build_one(args_tuple):
    (k_idx, cache_dir, overwrite) = args_tuple
    scan_all_data(root_data_path)
    start = time.time()
    cache_path = get_graph_cache_path(k_idx, cache_dir)
    if os.path.exists(cache_path) and (not overwrite):
        return {'k_idx': int(k_idx), 'status': 'exists', 'path': cache_path, 'elapsed_sec': 0.0}
    data = build_graph_data(k_idx)
    (cache_path, written) = save_graph_data_cache(k_idx, data=data, cache_dir=cache_dir, overwrite=overwrite)
    result = {'k_idx': int(k_idx), 'status': 'written' if written else 'exists', 'path': cache_path, 'elapsed_sec': time.time() - start}
    result.update(_data_stats(data))
    return result

def _select_indices(valid_indices, args):
    indices = valid_indices
    if args.start_k is not None:
        indices = [k for k in indices if k >= args.start_k]
    if args.end_k is not None:
        indices = [k for k in indices if k <= args.end_k]
    if args.num_shards > 1:
        indices = [k for (pos, k) in enumerate(indices) if pos % args.num_shards == args.shard_rank]
    if args.limit is not None:
        indices = indices[:args.limit]
    return indices

def parse_args():
    parser = argparse.ArgumentParser(description='Precompute and save offline PyG AMG graph cache files.')
    parser.add_argument('--cache-dir', default=None, help='Override graph cache directory.')
    parser.add_argument('--overwrite', action='store_true', help='Rebuild existing cache files.')
    parser.add_argument('--workers', type=int, default=1, help='Number of build worker processes.')
    parser.add_argument('--start-k', type=int, default=None, help='First k_idx to build.')
    parser.add_argument('--end-k', type=int, default=None, help='Last k_idx to build.')
    parser.add_argument('--limit', type=int, default=None, help='Build at most this many samples.')
    parser.add_argument('--num-shards', type=int, default=1, help='Total number of manual shards.')
    parser.add_argument('--shard-rank', type=int, default=0, help='Shard rank in [0, num_shards).')
    parser.add_argument('--report-every', type=int, default=50, help='Progress print interval.')
    return parser.parse_args()

def main():
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError('--num-shards must be >= 1')
    if args.shard_rank < 0 or args.shard_rank >= args.num_shards:
        raise ValueError('--shard-rank must be in [0, num_shards)')
    (data_mapping, n_total) = scan_all_data(root_data_path)
    valid_indices = [k for k in range(1, n_total + 1) if k in data_mapping]
    indices = _select_indices(valid_indices, args)
    cache_dir = get_graph_cache_dir(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    print('🚀 开始离线生成 PyG 图缓存')
    print(f'   数据根目录: {root_data_path}')
    print(f'   缓存目录: {cache_dir}')
    print(f'   总样本数: {len(valid_indices)} | 本次处理: {len(indices)}')
    print(f'   overwrite={args.overwrite} | workers={args.workers}')
    if args.num_shards > 1:
        print(f'   shard: {args.shard_rank}/{args.num_shards}')
    start_all = time.time()
    results = []
    failed = []
    written = 0
    skipped = 0
    if args.workers <= 1:
        iterator = enumerate(indices, start=1)
        for (count, k_idx) in iterator:
            try:
                result = _build_one((k_idx, cache_dir, args.overwrite))
                results.append(result)
                if result['status'] == 'written':
                    written += 1
                else:
                    skipped += 1
            except Exception as exc:
                failed.append({'k_idx': int(k_idx), 'error': repr(exc), 'traceback': traceback.format_exc()})
            if count % max(1, args.report_every) == 0 or count == len(indices):
                print(f'[Progress] {count}/{len(indices)} | written={written} skipped={skipped} failed={len(failed)}')
    else:
        tasks = [(k_idx, cache_dir, args.overwrite) for k_idx in indices]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_build_one, task): task[0] for task in tasks}
            for (count, future) in enumerate(as_completed(futures), start=1):
                k_idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if result['status'] == 'written':
                        written += 1
                    else:
                        skipped += 1
                except Exception as exc:
                    failed.append({'k_idx': int(k_idx), 'error': repr(exc), 'traceback': traceback.format_exc()})
                if count % max(1, args.report_every) == 0 or count == len(indices):
                    print(f'[Progress] {count}/{len(indices)} | written={written} skipped={skipped} failed={len(failed)}')
    elapsed = time.time() - start_all
    report = {'cache_dir': cache_dir, 'root_data_path': root_data_path, 'total_available': len(valid_indices), 'total_selected': len(indices), 'written': written, 'skipped': skipped, 'failed': len(failed), 'elapsed_sec': elapsed, 'results': results, 'failed_items': failed}
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(cache_dir, f'graph_cache_report_{timestamp}.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    if failed:
        failed_path = os.path.join(cache_dir, f'graph_cache_failed_{timestamp}.txt')
        with open(failed_path, 'w', encoding='utf-8') as f:
            for item in failed:
                f.write(f"k_idx={item['k_idx']} error={item['error']}\n")
                f.write(item['traceback'])
                f.write('\n' + '=' * 80 + '\n')
        print(f'⚠️  失败列表已保存: {failed_path}')
    print('\n==================================================')
    print('📊 PyG 图缓存生成完成')
    print(f'写入: {written} | 已存在跳过: {skipped} | 失败: {len(failed)}')
    print(f'耗时: {elapsed:.1f}s')
    print(f'报告: {report_path}')
    print('==================================================')
if __name__ == '__main__':
    main()
