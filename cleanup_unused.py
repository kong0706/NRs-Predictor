"""
cleanup_unused.py — 删除 model_ml_construct_list.py 和 model_graph_merge_hyper_list.py
生成的、app.py 用不到的文件/文件夹/模型。

Usage:
    python cleanup_unused.py --dry-run      # 仅预览，不删除（默认）
    python cleanup_unused.py --execute      # 执行删除
    python cleanup_unused.py --execute --keep-unused-models  # 删除中间文件但保留所有模型文件
"""

import os
import sys
import shutil
import argparse
import pandas as pd

# ============================================================
# 路径配置（与 app.py 完全一致）
# ============================================================
ML_BASELINE_BASE = './NURA_Baseline_ml'
GNN_BASELINE_BASE = './NURA_Baseline_gnn'
CONSENSUS_BASE = './NURA_consensus'

ML_TOP_K = 6
GNN_TOP_K = 3
NUM_REPLICATES = 5

# 所有 ML 模型与特征表示
ML_MODELS = ['lgb', 'RF', 'SVM', 'xgb']
ML_REPS = ['descriptors', 'maccs', 'morgan', 'rdk', 'mol2vec']

# 所有 GNN 模型
GNN_MODELS = ['GT', 'GIN', 'GCN', 'GAT', 'AFP']

# CONSENSUS_CONFIGS（与 app.py 完全一致）
CONSENSUS_CONFIGS = {
    ('FXR', 'binder'): ('none', 'none'),
    ('PPARD', 'binder'): ('none', 'none'),
    ('RXR', 'binder'): ('none', 'none'),
    ('ERB', 'binder'): ('none', 'none'),
    ('PR', 'binder'): ('none', 'none'),
    ('AR', 'binder'): ('none', 'none'),
    ('ERA', 'binder'): ('none', 'none'),
    ('PXR', 'binder'): ('none', 'none'),
    ('PPARG', 'binder'): ('none', 'none'),
    ('GR', 'binder'): ('none', 'none'),
    ('ERA', 'antagonist'): ('none', 'none'),
    ('GR', 'antagonist'): ('none', 'none'),
    ('AR', 'antagonist'): ('none', 'none'),
    ('PR', 'antagonist'): ('none', 'none'),
    ('PPARG', 'antagonist'): ('rus', 'ros'),
    ('FXR', 'antagonist'): ('rus', 'ros'),
    ('ERB', 'antagonist'): ('rus', 'ros'),
    ('AR', 'agonist'): ('none', 'none'),
    ('PPARD', 'agonist'): ('none', 'none'),
    ('GR', 'agonist'): ('none', 'none'),
    ('ERA', 'agonist'): ('none', 'none'),
    ('PPARG', 'agonist'): ('none', 'none'),
    ('PXR', 'agonist'): ('none', 'none'),
    ('RXR', 'agonist'): ('rus', 'ros'),
    ('ERB', 'agonist'): ('rus', 'ros'),
    ('PR', 'agonist'): ('rus', 'ros'),
    ('FXR', 'agonist'): ('rus', 'ros'),
}


def get_top_ml_models(target, mode, ml_sampling, k=ML_TOP_K, metric='MCC'):
    """
    读取 results_mean.xlsx / results_std.xlsx，返回 Top-K ML 模型配置列表。
    与 app.py get_top_ml_models() 逻辑一致。
    """
    ml_base_dir = f'{ML_BASELINE_BASE}/{target}/{mode}/{ml_sampling}'
    mean_path = os.path.join(ml_base_dir, 'results_10to1', 'results_mean.xlsx')
    std_path = os.path.join(ml_base_dir, 'results_10to1', 'results_std.xlsx')

    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        return None  # 尚未训练

    df_mean = pd.read_excel(mean_path, sheet_name='test')
    df_std = pd.read_excel(std_path, sheet_name='test')

    df_combined = pd.DataFrame({
        'model': df_mean['model'],
        'rep': df_mean['rep'],
        f'{metric}_mean': df_mean[metric],
        f'{metric}_std': df_std[metric]
    })
    df_combined['lower_bound_score'] = (
        df_combined[f'{metric}_mean'] - df_combined[f'{metric}_std']
    )
    df_sorted = df_combined.sort_values(by='lower_bound_score', ascending=False)
    top_k_df = df_sorted.head(k)
    return list(zip(top_k_df['model'], top_k_df['rep']))


def get_top_gnn_models(target, mode, gnn_sampling, k=GNN_TOP_K, metric='MCC'):
    """
    读取 results_mean.xlsx / results_std.xlsx，返回 Top-K GNN 模型名称列表。
    与 app.py get_top_gnn_models() 逻辑一致。
    """
    gnn_base_dir = f'{GNN_BASELINE_BASE}/{target}/{mode}/{gnn_sampling}'
    task_dir = os.path.join(gnn_base_dir, 'graph_results', 'results_mean_std')
    mean_path = os.path.join(task_dir, 'results_mean.xlsx')
    std_path = os.path.join(task_dir, 'results_std.xlsx')

    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        return None  # 尚未训练

    df_mean = pd.read_excel(mean_path, sheet_name='Test')
    df_std = pd.read_excel(std_path, sheet_name='Test')

    df_combined = pd.DataFrame({
        'model': df_mean['Model'],
        f'{metric}_mean': df_mean[metric],
        f'{metric}_std': df_std[metric]
    })
    df_combined['lower_bound_score'] = (
        df_combined[f'{metric}_mean'] - df_combined[f'{metric}_std']
    )
    df_sorted = df_combined.sort_values(by='lower_bound_score', ascending=False)
    top_k_df = df_sorted.head(k)
    return top_k_df['model'].tolist()


def collect_cleanup_tasks():
    """
    扫描所有任务配置，收集待清理项。
    返回 (dirs_to_delete, files_to_delete) 两个列表。
    """
    dirs_to_delete = []   # 整个目录删除
    files_to_delete = []  # 单个文件删除

    for (target, mode), (ml_sampling, gnn_sampling) in CONSENSUS_CONFIGS.items():
        # ============================================================
        # ML 中间产物
        # ============================================================
        ml_base = os.path.join(ML_BASELINE_BASE, target, mode, ml_sampling)

        # ① results_all/ — 各 replicate 独立 Excel，app.py 只读 results_10to1/
        results_all_dir = os.path.join(ml_base, 'results_all')
        if os.path.isdir(results_all_dir):
            dirs_to_delete.append((results_all_dir, 'ML per-replicate intermediate results'))

        # ② cv_results/ — GridSearchCV 详细结果，app.py 完全不读
        cv_results_dir = os.path.join(ml_base, 'cv_results')
        if os.path.isdir(cv_results_dir):
            dirs_to_delete.append((cv_results_dir, 'ML GridSearchCV detail files'))

        # ③ results_mean±std.xlsx — 人类可读格式，app.py 读 mean/std 两个独立文件
        mean_std_path = os.path.join(ml_base, 'results_10to1', 'results_mean±std.xlsx')
        if os.path.isfile(mean_std_path):
            files_to_delete.append((mean_std_path, 'ML human-readable mean±std (app reads separate mean/std files)'))

        # ============================================================
        # GNN 中间产物
        # ============================================================
        gnn_base = os.path.join(GNN_BASELINE_BASE, target, mode, gnn_sampling)
        gnn_results_dir = os.path.join(gnn_base, 'graph_results')

        # ④ graph_results/results_all/ — 各 replicate 独立 Excel
        gnn_results_all = os.path.join(gnn_results_dir, 'results_all')
        if os.path.isdir(gnn_results_all):
            dirs_to_delete.append((gnn_results_all, 'GNN per-replicate intermediate results'))

        # ⑤ results_mean±std.xlsx — 人类可读格式
        gnn_mean_std = os.path.join(gnn_results_dir, 'results_mean_std', 'results_mean±std.xlsx')
        if os.path.isfile(gnn_mean_std):
            files_to_delete.append((gnn_mean_std, 'GNN human-readable mean±std (app reads separate mean/std files)'))

        # ============================================================
        # Consensus 中间产物
        # ============================================================
        consensus_dir = os.path.join(
            CONSENSUS_BASE, target, mode,
            f'consensus_stacking_{ml_sampling}_{gnn_sampling}'
        )

        # ⑥ results_{1..5}.xlsx — 各 replicate 独立评估结果，app.py 不读
        for rep in range(1, NUM_REPLICATES + 1):
            res_path = os.path.join(consensus_dir, f'results_{rep}.xlsx')
            if os.path.isfile(res_path):
                files_to_delete.append((res_path, 'Consensus per-replicate result (app only loads stacking_model_*.joblib)'))

        # ============================================================
        # 未使用的模型文件（根据 Top-K 选择）
        # ============================================================
        top_ml = get_top_ml_models(target, mode, ml_sampling, k=ML_TOP_K)
        top_gnn = get_top_gnn_models(target, mode, gnn_sampling, k=GNN_TOP_K)

        if top_ml is not None:
            # 构建 Top-K 集合：{(model_name, feat_type), ...}
            top_ml_set = set(top_ml)
            for rep in range(1, NUM_REPLICATES + 1):
                model_rep_dir = os.path.join(ml_base, 'final_models', f'Replicate_{rep}')
                if not os.path.isdir(model_rep_dir):
                    continue
                for model_name in ML_MODELS:
                    for feat_type in ML_REPS:
                        if (model_name, feat_type) not in top_ml_set:
                            fpath = os.path.join(model_rep_dir, f'{model_name}_{feat_type}.joblib')
                            if os.path.isfile(fpath):
                                files_to_delete.append((
                                    fpath,
                                    f'ML model not in Top-{ML_TOP_K} (MCC) for {target}-{mode}'
                                ))

        if top_gnn is not None:
            top_gnn_set = set(top_gnn)
            for rep in range(1, NUM_REPLICATES + 1):
                model_rep_dir = os.path.join(gnn_base, 'graph_models', f'Replicate_{rep}')
                if not os.path.isdir(model_rep_dir):
                    continue
                for model_name in GNN_MODELS:
                    if model_name not in top_gnn_set:
                        fpath = os.path.join(model_rep_dir, f'{model_name}.pth')
                        if os.path.isfile(fpath):
                            files_to_delete.append((
                                fpath,
                                f'GNN model not in Top-{GNN_TOP_K} (MCC) for {target}-{mode}'
                            ))

    return dirs_to_delete, files_to_delete


def format_size(size_bytes):
    """将字节数转为可读大小。"""
    for unit in ('B', 'KB', 'MB', 'GB'):
        if size_bytes < 1024:
            return f'{size_bytes:.1f} {unit}'
        size_bytes /= 1024
    return f'{size_bytes:.1f} TB'


def dir_size(path):
    """递归计算目录总大小。"""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def main():
    parser = argparse.ArgumentParser(
        description='清理 app.py 用不到的训练中间产物与模型文件'
    )
    parser.add_argument(
        '--execute', action='store_true',
        help='执行删除（默认 dry-run 仅预览）'
    )
    parser.add_argument(
        '--keep-unused-models', action='store_true',
        help='保留所有模型文件，仅删除中间结果目录/文件'
    )
    args = parser.parse_args()

    dirs_to_delete, files_to_delete = collect_cleanup_tasks()

    # 过滤：如果指定了 --keep-unused-models，去掉模型文件
    if args.keep_unused_models:
        files_to_delete = [
            (path, reason) for path, reason in files_to_delete
            if 'not in Top-' not in reason
        ]

    # ============================================================
    # 统计大小
    # ============================================================
    total_dir_size = 0
    for dpath, _ in dirs_to_delete:
        if os.path.isdir(dpath):
            total_dir_size += dir_size(dpath)

    total_file_size = 0
    for fpath, _ in files_to_delete:
        if os.path.isfile(fpath):
            total_file_size += os.path.getsize(fpath)

    total_size = total_dir_size + total_file_size

    # ============================================================
    # 输出预览
    # ============================================================
    mode_str = 'DRY-RUN (Preview)' if not args.execute else 'EXECUTE (Deleting)'
    print('=' * 80)
    print(f'  Cleanup Script — {mode_str}')
    print('=' * 80)

    # --- 目录 ---
    print(f'\n📁 Directories to remove ({len(dirs_to_delete)} items, {format_size(total_dir_size)}):')
    print('-' * 80)
    if dirs_to_delete:
        for dpath, reason in sorted(dirs_to_delete):
            sz = dir_size(dpath) if os.path.isdir(dpath) else 0
            print(f'  [{format_size(sz):>10}]  {dpath}')
            print(f'               ↳ {reason}')
    else:
        print('  (none)')

    # --- 文件 ---
    print(f'\n📄 Files to remove ({len(files_to_delete)} items, {format_size(total_file_size)}):')
    print('-' * 80)
    if files_to_delete:
        for fpath, reason in sorted(files_to_delete):
            sz = os.path.getsize(fpath) if os.path.isfile(fpath) else 0
            print(f'  [{format_size(sz):>10}]  {fpath}')
            print(f'               ↳ {reason}')
    else:
        print('  (none)')

    print('\n' + '=' * 80)
    print(f'  Total: {len(dirs_to_delete)} dirs + {len(files_to_delete)} files = {format_size(total_size)}')
    print('=' * 80)

    if not args.execute:
        print('\n💡 This is a DRY-RUN. Add --execute to actually delete.')
        if not args.keep_unused_models:
            print('💡 Add --keep-unused-models to keep all model files and only clean intermediate artifacts.')
        return

    # ============================================================
    # 执行删除
    # ============================================================
    print('\n🚀 Executing cleanup...\n')

    # 先删文件
    for fpath, reason in files_to_delete:
        try:
            os.remove(fpath)
            print(f'  ✓ Deleted file: {fpath}')
        except OSError as e:
            print(f'  ✗ Failed to delete {fpath}: {e}')

    # 再删目录
    for dpath, reason in dirs_to_delete:
        try:
            shutil.rmtree(dpath)
            print(f'  ✓ Deleted dir:  {dpath}')
        except OSError as e:
            print(f'  ✗ Failed to delete {dpath}: {e}')

    # 清理空父目录
    empty_dirs_removed = 0
    for base in [ML_BASELINE_BASE, GNN_BASELINE_BASE, CONSENSUS_BASE]:
        if not os.path.isdir(base):
            continue
        for dirpath, dirnames, filenames in os.walk(base, topdown=False):
            if dirpath == base:
                continue
            if not os.listdir(dirpath):
                try:
                    os.rmdir(dirpath)
                    empty_dirs_removed += 1
                except OSError:
                    pass

    if empty_dirs_removed:
        print(f'\n  ✓ Removed {empty_dirs_removed} empty parent directories.')

    print('\n✅ Cleanup complete.')


if __name__ == '__main__':
    main()
