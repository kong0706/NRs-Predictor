import streamlit as st
st.set_page_config(page_title="NURA Activity Predictor", layout="wide")

import pandas as pd
import numpy as np
import os
import json
import torch
import warnings
import sys
from joblib import load
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, Draw

import deepchem as dc
from streamlit_ketcher import st_ketcher
from torch_geometric.loader import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from utils import mol_to_graph_data_obj_simple
from model import GINModel, GCNModel, GraphTransformerModel, GATModel
from torch_geometric.nn.models import AttentiveFP

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PATH CONFIGURATION
PRETRAIN_MOL2VEC_PATH = './model_300dim.pkl'
HYPERPARAMS_DIR = './best_hyperparameters'

# Consensus stacking paths (from Consensus_Model_stacking_auto.py)
ML_BASELINE_BASE = './NURA_Baseline_ml'
GNN_BASELINE_BASE = './NURA_Baseline_gnn'
CONSENSUS_BASE = './NURA_consensus'

ML_TOP_K = 5     # Number of top ML models to select
GNN_TOP_K = 2    # Number of top GNN models to select
NUM_REPLICATES = 5  # Total replicates from Consensus_Model_stacking_auto.py

# GNN MODEL CLASS MAP
GNN_CLASS_MAP = {
    "GT": GraphTransformerModel,
    "GIN": GINModel,
    "GCN": GCNModel,
    "GAT": GATModel,
    "AFP": AttentiveFP
}

# CONSENSUS TASK CONFIGURATION
# Derived from Consensus_Model_stacking_auto.py CONFIGS
# (target, mode) -> (ml_sampling, gnn_sampling)
CONSENSUS_CONFIGS = {
    # --- Binders (ml_sampling='none', gnn_sampling='none') ---
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

    # --- Antagonists (ml_sampling='none', gnn_sampling='none') ---
    ('ERA', 'antagonist'): ('none', 'none'),
    ('GR', 'antagonist'): ('none', 'none'),
    ('AR', 'antagonist'): ('none', 'none'),
    ('PR', 'antagonist'): ('none', 'none'),

    # --- Antagonists (ml_sampling='rus', gnn_sampling='ros') ---
    ('PPARG', 'antagonist'): ('rus', 'ros'),
    ('FXR', 'antagonist'): ('rus', 'ros'),
    ('ERB', 'antagonist'): ('rus', 'ros'),

    # --- Agonists (ml_sampling='none', gnn_sampling='none') ---
    ('AR', 'agonist'): ('none', 'none'),
    ('PPARD', 'agonist'): ('none', 'none'),
    ('GR', 'agonist'): ('none', 'none'),
    ('ERA', 'agonist'): ('none', 'none'),
    ('PPARG', 'agonist'): ('none', 'none'),
    ('PXR', 'agonist'): ('none', 'none'),

    # --- Agonists (ml_sampling='rus', gnn_sampling='ros') ---
    ('RXR', 'agonist'): ('rus', 'ros'),
    ('ERB', 'agonist'): ('rus', 'ros'),
    ('PR', 'agonist'): ('rus', 'ros'),
    ('FXR', 'agonist'): ('rus', 'ros'),
}

# AD cache
AD_CACHE = {}
# Cache for top-k model selections
_TOP_MODEL_CACHE = {}

# TOP-K MODEL SELECTION (from Consensus_Model_stacking_auto.py)
def get_top_ml_models(target, mode, ml_sampling, k=ML_TOP_K, metric='MCC'):
    """
    Select top-k ML models based on test set (mean MCC - std MCC).
    Reads results_mean.xlsx / results_std.xlsx from the ML baseline directory.
    """
    cache_key = ('ml', target, mode, ml_sampling, k, metric)
    if cache_key in _TOP_MODEL_CACHE:
        return _TOP_MODEL_CACHE[cache_key]

    ml_base_dir = f'{ML_BASELINE_BASE}/{target}/{mode}/{ml_sampling}'
    mean_path = os.path.join(ml_base_dir, 'results_10to1', 'results_mean.xlsx')
    std_path = os.path.join(ml_base_dir, 'results_10to1', 'results_std.xlsx')

    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(
            f"ML results not found for {target}-{mode} (sampling={ml_sampling}). "
            f"Expected: {mean_path}"
        )

    df_mean = pd.read_excel(mean_path, sheet_name='test')
    df_std = pd.read_excel(std_path, sheet_name='test')

    df_combined = pd.DataFrame({
        'model': df_mean['model'],
        'rep': df_mean['rep'],
        f'{metric}_mean': df_mean[metric],
        f'{metric}_std': df_std[metric]
    })
    df_combined['lower_bound_score'] = df_combined[f'{metric}_mean'] - df_combined[f'{metric}_std']
    df_sorted = df_combined.sort_values(by='lower_bound_score', ascending=False)

    top_k_df = df_sorted.head(k)
    ensemble_configs = list(zip(top_k_df['model'], top_k_df['rep']))

    _TOP_MODEL_CACHE[cache_key] = ensemble_configs
    return ensemble_configs


def get_top_gnn_models(target, mode, gnn_sampling, k=GNN_TOP_K, metric='MCC'):
    """
    Select top-k GNN models based on test set (mean MCC - std MCC).
    Reads results_mean.xlsx / results_std.xlsx from the GNN baseline directory.
    """
    cache_key = ('gnn', target, mode, gnn_sampling, k, metric)
    if cache_key in _TOP_MODEL_CACHE:
        return _TOP_MODEL_CACHE[cache_key]

    gnn_base_dir = f'{GNN_BASELINE_BASE}/{target}/{mode}/{gnn_sampling}'
    task_dir = os.path.join(gnn_base_dir, "graph_results", "results_mean_std")
    mean_path = os.path.join(task_dir, 'results_mean.xlsx')
    std_path = os.path.join(task_dir, 'results_std.xlsx')

    if not os.path.exists(mean_path) or not os.path.exists(std_path):
        raise FileNotFoundError(
            f"GNN results not found for {target}-{mode} (sampling={gnn_sampling}). "
            f"Expected: {mean_path}"
        )

    df_mean = pd.read_excel(mean_path, sheet_name='Test')
    df_std = pd.read_excel(std_path, sheet_name='Test')

    df_combined = pd.DataFrame({
        'model': df_mean['Model'],
        f'{metric}_mean': df_mean[metric],
        f'{metric}_std': df_std[metric]
    })
    df_combined['lower_bound_score'] = df_combined[f'{metric}_mean'] - df_combined[f'{metric}_std']
    df_sorted = df_combined.sort_values(by='lower_bound_score', ascending=False)

    top_k_df = df_sorted.head(k)
    gnn_models = top_k_df['model'].tolist()

    _TOP_MODEL_CACHE[cache_key] = gnn_models
    return gnn_models

def calculate_features(smiles_list, tag):
    """计算配体的分子特征"""
    tag = tag.lower()
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    if "descriptors" in tag or "descriptor" in tag:
        return np.array([[desc(mol) for n, desc in Descriptors.descList] for mol in mols])
    elif "maccs" in tag:
        return np.array([list(MACCSkeys.GenMACCSKeys(mol)) for mol in mols])
    elif "rdk" in tag:
        return np.array([list(Chem.RDKFingerprint(mol)) for mol in mols])
    elif "morgan" in tag:
        return np.array([list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)) for mol in mols])
    elif "mol2vec" in tag:
        featurizer = dc.feat.Mol2VecFingerprint(pretrain_model_path=PRETRAIN_MOL2VEC_PATH)
        return np.array([featurizer.featurize(s)[0].tolist() for s in smiles_list])
    return None

def clean_smiles_list(smiles_list):
    """标准化分子SMILES"""
    new_list = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol:
            s2 = Chem.MolToSmiles(mol, isomericSmiles=False)
            new_list.append(s2)
        else:
            new_list.append(s)
    return new_list

# GNN MODEL LOADING
def load_gnn_model(model_name, target, mode, gnn_sampling, replicate):
    """
    Load a GNN model using the consensus path structure.
    Path: NURA_Baseline_gnn/{target}/{mode}/{gnn_sampling}/graph_models/Replicate_{rep}/{model_name}.pth
    """
    json_path = os.path.join(HYPERPARAMS_DIR, f"{target}_best_hyperparameters.json")
    pth_path = os.path.join(
        GNN_BASELINE_BASE, target, mode, gnn_sampling,
        "graph_models", f"Replicate_{replicate}", f"{model_name}.pth"
    )

    with open(json_path, 'r') as f:
        all_params = json.load(f)

    study_name = f"{mode}_{gnn_sampling}_{model_name}"
    if study_name not in all_params:
        raise KeyError(f"Hyperparameter key '{study_name}' not found in {json_path}")
    params = all_params[study_name]

    model_class = GNN_CLASS_MAP[model_name]

    if model_class == GraphTransformerModel:
        model = model_class(in_channels=32, hidden_channels=params['hidden_channels'], out_channels=1,
                            edge_dim=11, num_layers=params['num_layers'], dropout=params['dropout'],
                            n_heads=params['n_heads'])
    elif model_class == AttentiveFP:
        model = model_class(in_channels=32, hidden_channels=params['hidden_channels'], out_channels=1,
                            edge_dim=11, num_layers=params['num_layers'], dropout=params['dropout'],
                            num_timesteps=params['num_timesteps'])
    else:
        model = model_class(in_channels=32, hidden_channels=params['hidden_channels'], out_channels=1,
                            edge_dim=11, num_layers=params['num_layers'], dropout=params['dropout'])

    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# STACKING PREDICTION (Consensus methodology — 5-replicate ensemble averaging)
def run_stacking_prediction(target, mode, smiles_list, ml_sampling, gnn_sampling):
    """
    Run prediction using the consensus stacking pipeline:
    1. Select top-k ML and top-k GNN models by test MCC (shared across all replicates)
    2. For each of the 5 random-seed replicates:
       a. Load that replicate's base models and stacking meta-model
       b. Generate base predictions from each sub-model
       c. Apply the LogisticRegressionCV meta-model for that replicate's final probability
    3. Average the final probabilities across all 5 replicates
    """
    smiles_list = clean_smiles_list(smiles_list)

    # 1. Get top-k model configurations (same for all replicates)
    try:
        ml_ensemble = get_top_ml_models(target, mode, ml_sampling, k=ML_TOP_K)
        gnn_models = get_top_gnn_models(target, mode, gnn_sampling, k=GNN_TOP_K)
    except FileNotFoundError as e:
        st.error(f"Consensus results not found for {target}-{mode}: {e}")
        return None, None

    consensus_dir = os.path.join(
        CONSENSUS_BASE, target, mode,
        f"consensus_stacking_{ml_sampling}_{gnn_sampling}"
    )

    all_rep_final_probs = []  # store final probabilities from each replicate

    # 2. Pre-compute GNN graph data (same across replicates)
    graph_data = [mol_to_graph_data_obj_simple(Chem.MolFromSmiles(s)) for s in smiles_list]
    loader = DataLoader(graph_data, batch_size=len(smiles_list))

    for rep in range(1, NUM_REPLICATES + 1):
        # 2a. Load stacking meta-model for this replicate
        meta_model_path = os.path.join(consensus_dir, f"stacking_model_{rep}.joblib")
        if not os.path.exists(meta_model_path):
            st.warning(f"Stacking meta-model missing for replicate {rep}: {meta_model_path}")
            continue

        meta_model = load(meta_model_path)
        all_probs = []

        # 2b. Generate ML base predictions for this replicate
        for model_name, feat_type in ml_ensemble:
            model_path = os.path.join(
                ML_BASELINE_BASE, target, mode, ml_sampling,
                "final_models", f"Replicate_{rep}",
                f"{model_name}_{feat_type}.joblib"
            )
            if not os.path.exists(model_path):
                continue

            base_model = load(model_path)
            X = calculate_features(smiles_list, feat_type)
            if X is not None:
                all_probs.append(base_model.predict_proba(X)[:, 1])

        # 2c. Generate GNN base predictions for this replicate
        for model_name in gnn_models:
            try:
                model = load_gnn_model(model_name, target, mode, gnn_sampling, rep)
                with torch.no_grad():
                    for data in loader:
                        data = data.to(device)
                        out = model(data.x.float(), data.edge_index.long(),
                                   data.edge_attr.float(), data.batch.long())
                        all_probs.append(torch.sigmoid(out).cpu().numpy().flatten())
            except Exception:
                continue

        if not all_probs:
            continue

        # 2d. Stack predictions and apply meta-model for this replicate
        X_meta = np.column_stack(all_probs)
        rep_final_probs = meta_model.predict_proba(X_meta)[:, 1]
        all_rep_final_probs.append(rep_final_probs)

    if not all_rep_final_probs:
        st.error(f"No replicate predictions generated for {target}-{mode}")
        return None, None

    # 3. Average final probabilities across all replicates
    all_rep_final_probs = np.array(all_rep_final_probs)  # shape: (n_valid_reps, n_samples)
    final_probs = np.mean(all_rep_final_probs, axis=0)

    preds = (final_probs >= 0.5).astype(int)
    return preds, final_probs

# PREDICTION ENTRY POINT
def run_prediction(target, mode, smiles_list):
    """
    Consensus stacking prediction with 5-replicate ensemble averaging:
    1. Select top-k ML and top-k GNN models by test set MCC
    2. For each of the 5 replicates, generate base predictions and apply
       the LogisticRegressionCV meta-model for that replicate's final probability
    3. Average the final probabilities across all 5 replicates
    """
    config = CONSENSUS_CONFIGS.get((target, mode))
    if config is None:
        st.error(f"No consensus configuration for {target}-{mode}")
        return None, None

    ml_sampling, gnn_sampling = config
    return run_stacking_prediction(target, mode, smiles_list, ml_sampling, gnn_sampling)

@st.cache_data
def load_threshold():
    """加载每个训练集计算得到的95%百分位数阈值"""
    df = pd.read_csv("similarity_threshold.csv")
    return {(row['receptor'], row['train_type']): row['p95_similarity'] for _, row in df.iterrows()}

threshold_dict = load_threshold()

def smiles_to_fp(smiles_list, nBits=2048):
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits)
            # 转为numpy 0/1数组
            arr = np.zeros(nBits, dtype=np.int8)
            for bit in fp.GetOnBits():
                arr[bit] = 1
            fps.append(arr)
            valid_idx.append(i)
    return np.array(fps), valid_idx

def calculate_ad(test_fps, train_fps, threshold, k=5):
    n_test = test_fps.shape[0]

    # 交集矩阵: (n_test, n_train)
    intersection = test_fps @ train_fps.T

    # 每个分子的置1位数
    test_sums = test_fps.sum(axis=1)      # (n_test,)
    train_sums = train_fps.sum(axis=1)    # (n_train,)

    # 并集矩阵: (n_test, n_train)
    union = test_sums[:, None] + train_sums[None, :] - intersection

    # Tanimoto = 交集 / 并集
    with np.errstate(invalid='ignore'):
        sims = intersection / union
    sims = np.nan_to_num(sims, nan=0.0)

    # 对每个测试分子取 top-k 平均
    results = {}
    for idx in range(n_test):
        row_sims = sims[idx]
        if len(row_sims) >= k:
            top_k = np.partition(row_sims, -k)[-k:]
        else:
            top_k = row_sims
        avg_sim = top_k.mean()
        results[idx] = 'Inside AD' if avg_sim >= threshold else 'Outside AD'

    return results

def load_train_fps(receptor, train_type):
    """加载训练集分子指纹（全量数据，numpy数组）"""
    key = (receptor, train_type)

    if key in AD_CACHE:
        return AD_CACHE[key]

    path = os.path.join("datasets", receptor, train_type, "morgan", "morgan.csv")
    df = pd.read_csv(path)

    # 使用全部数据，直接转为numpy数组
    fps = df.iloc[:, 1:-1].values.astype(np.int8)

    AD_CACHE[key] = fps
    return fps

def run_ad(smiles_list, receptor, mode):
    """计算AD"""
    train_type = f"{mode}_train"
    threshold = threshold_dict.get((receptor, train_type))

    if threshold is None:
        return ["-"] * len(smiles_list)

    test_fps, valid_idx = smiles_to_fp(smiles_list)
    train_fps = load_train_fps(receptor, train_type)
    ad_raw = calculate_ad(test_fps, train_fps, threshold)
    idx_map = {orig: i for i, orig in enumerate(valid_idx)}

    results = []
    for i in range(len(smiles_list)):
        if i in idx_map:
            results.append(ad_raw[idx_map[i]])
        else:
            results.append("Invalid SMILES")

    return results

# STREAMLIT UI
def main():
    st.title("Nuclear Receptor Activity Prediction Platform")
    st.image("Schematic diagram.png", caption="Schematic Diagram", use_column_width=True)

    # All tasks from consensus configuration
    ALL_TASKS = sorted(CONSENSUS_CONFIGS.keys())

    # Sidebar
    st.sidebar.header("Target Configuration")
    all_targets = sorted(list(set([k[0] for k in ALL_TASKS])))
    selected_target = st.sidebar.selectbox("Select Receptor", all_targets)

    available_modes = sorted(list(set(
        [k[1] for k in ALL_TASKS if k[0] == selected_target]
    )))
    selected_mode = st.sidebar.selectbox("Select Mode", available_modes)

    # Input area
    input_type = st.radio("Input Method", ["Draw Molecule", "SMILES String", "Batch CSV Upload"])
    smiles_list = []

    if input_type == "Draw Molecule":
        drawn = st_ketcher("")
        if drawn:
            st.write(f"Generated SMILES: {drawn}")
            smiles_list = [drawn]

    elif input_type == "SMILES String":
        s_input = st.text_input("Enter SMILES")
        if s_input:
            smiles_list = [s_input.strip()]

    else:
        file = st.file_uploader("Upload CSV", type=["csv"])
        if file:
            df = pd.read_csv(file)
            if "SMILES" in df.columns:
                smiles_list = df["SMILES"].dropna().tolist()

    # Buttons
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

    with btn_col2:
        start_single = st.button("Start Calculation", use_container_width=True)
        start_all = st.button("Run All Targets", use_container_width=True)

    # Single task prediction
    if start_single and smiles_list:
        with st.spinner("Calculating..."):
            preds, probs = run_prediction(selected_target, selected_mode, smiles_list)
            ad_results = run_ad(smiles_list, selected_target, selected_mode)

            if preds is not None:
                st.subheader("Results Table")
                res_df = pd.DataFrame({
                    "SMILES": smiles_list,
                    "Probability": [f"{p:.4f}" for p in probs],
                    "Outcome": ["Active" if p == 1 else "Inactive" for p in preds],
                    "Applicability Domain": ad_results
                })
                st.table(res_df)

                if len(smiles_list) == 1:
                    mol = Chem.MolFromSmiles(smiles_list[0])
                    if mol:
                        st.image(Draw.MolToImage(mol, size=(300, 300)))

    # All tasks prediction
    if start_all and smiles_list:
        st.warning("Running all targets may take several minutes.")

        with st.spinner("Running all targets..."):
            all_results = []
            progress = st.progress(0)
            total = len(ALL_TASKS)

            for i, (target, mode) in enumerate(ALL_TASKS):
                try:
                    preds, probs = run_prediction(target, mode, smiles_list)
                    ad_results = run_ad(smiles_list, target, mode)
                    if preds is None:
                        continue

                    for j, smi in enumerate(smiles_list):
                        all_results.append({
                            "SMILES": smi,
                            "Target": target,
                            "Mode": mode,
                            "Probability": probs[j],
                            "Outcome": "Active" if preds[j] == 1 else "Inactive",
                            "Applicability Domain": ad_results[j]
                        })

                    # Clear GPU cache
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    st.warning(f"{target}-{mode} failed: {e}")

                progress.progress((i + 1) / total)

            res_df = pd.DataFrame(all_results)

            # Results
            st.subheader("All Predictions")
            st.dataframe(res_df)

            # Download
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "all_predictions.csv", "text/csv")

if __name__ == "__main__":
    main()