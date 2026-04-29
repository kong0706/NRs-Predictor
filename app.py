import streamlit as st
st.set_page_config(page_title="NURA Activity Predictor", layout="wide")

import pandas as pd
import numpy as np
import os
import json
import torch
import warnings
import io
import sys
from joblib import load
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, Draw
import deepchem as dc
from streamlit_ketcher import st_ketcher
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from rdkit.DataStructs import ExplicitBitVect, BulkTanimotoSimilarity

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from utils import mol_to_graph_data_obj_simple
from model import GINModel, GCNModel, GraphTransformerModel
from torch_geometric.nn.models import AttentiveFP

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_PATH = "./NURA_ml"
PRETRAIN_MOL2VEC_PATH = './model_300dim.pkl'
HYPERPARAMS_DIR = './best_hyperparameters'
DL_WEIGHTS_BASE = './graph_models'

ENSEMBLE_TASKS = [
    ("PPARG", "antagonist"),
    ("FXR", "antagonist"),
    ("ERB", "antagonist"),
    ("ERA", "antagonist"),
    ("RXR", "agonist"),
    ("ERB", "agonist")
]

ENSEMBLE_SUBMODELS = {
    ("PPARG", "antagonist"): {
        "ml": [("ros", "xgb+rdk.joblib"), ("ros", "SVM+maccs.joblib"), ("ros", "xgb+descriptors.joblib"),
               ("rus", "SVM+maccs.joblib"), ("rus", "lgb+descriptors.joblib"), ("rus", "xgb+maccs.joblib")],
        "dl": ["GIN", "AFP", "GCN"]
    },
    ("FXR", "antagonist"): {
        "ml": [("ros", "lgb+descriptors.joblib"), ("ros", "lgb+rdk.joblib"), ("ros", "lgb+mol2vec.joblib"),
               ("ros", "SVM+rdk.joblib"), ("ros", "RF+rdk.joblib"), ("ros", "lgb+maccs.joblib")],
        "dl": ["GT", "AFP","GCN"]
    },
    ("ERB", "antagonist"): {
        "ml": [("ros", "lgb+descriptors.joblib"), ("ros", "SVM+morgan.joblib"), ("ros", "xgb+descriptors.joblib"),
               ("ros", "SVM+maccs.joblib"), ("ros", "lgb+morgan.joblib"), ("ros", "xgb+morgan.joblib")],
        "dl": ["GIN", "AFP","GCN"]
    },
    ("ERA", "antagonist"): {
        "ml": [("ros", "xgb+descriptors.joblib"), ("ros", "lgb+descriptors.joblib"), ("ros", "lgb+mol2vec.joblib"),
               ("ros", "xgb+mol2vec.joblib"), ("ros", "lgb+morgan.joblib"), ("ros", "xgb+maccs.joblib")],
        "dl": ["GT", "AFP","GCN"]
    },
    ("RXR", "agonist"): {
        "ml": [("ros", "RF+descriptors.joblib"), ("ros", "RF+rdk.joblib"), ("ros", "lgb+mol2vec.joblib"),
               ("rus", "SVM+morgan.joblib"), ("ros", "RF+mol2vec.joblib"), ("rus", "lgb+descriptors.joblib")],
        "dl": ["GT", "AFP", "GCN"]
    },
    ("ERB", "agonist"): {
        "ml": [("ros", "lgb+mol2vec.joblib"), ("ros", "SVM+mol2vec.joblib"), ("ros", "xgb+mol2vec.joblib"),
               ("ros", "SVM+morgan.joblib"), ("ros", "lgb+rdk.joblib"), ("ros", "lgb+morgan.joblib")],
        "dl": ["GT", "AFP","GIN"]
    }
}

SINGLE_MODEL_CONFIG = {
    ("PPARG", "agonist"): "SVM+rdk.joblib",
    ("PPARG", "binder"): "xgb+descriptors.joblib",
    ("FXR", "agonist"): "lgb+rdk.joblib",
    ("FXR", "binder"): "SVM+morgan.joblib",
    ("PXR", "agonist"): "lgb+descriptors.joblib",
    ("PXR", "binder"): "RF+maccs.joblib",
    ("GR", "agonist"): "lgb+descriptors.joblib",
    ("GR", "binder"): "SVM+maccs.joblib",
    ("GR", "antagonist"): "SVM+rdk.joblib",
    ("RXR", "binder"): "lgb+descriptors.joblib",
    ("ERB", "binder"): "SVM+rdk.joblib",
    ("AR", "agonist"): "SVM+mol2vec.joblib",
    ("AR", "binder"): "lgb+morgan.joblib",
    ("AR", "antagonist"): "lgb+descriptors.joblib",
    ("PPARD", "agonist"): "lgb+morgan.joblib",
    ("PPARD", "binder"): "SVM+rdk.joblib",
    ("PR", "agonist"): "RF+rdk.joblib",
    ("PR", "binder"): "lgb+morgan.joblib",
    ("PR", "antagonist"): "lgb+descriptors.joblib",
    ("ERA", "agonist"): "RF+descriptors.joblib",
    ("ERA", "binder"): "xgb+descriptors.joblib"
}

# AD缓存
AD_CACHE = {}

@st.cache_data
def load_threshold():
    """加载每个训练集计算得到的的95%百分位数阈值"""
    df = pd.read_csv("train_similarity_threshold.csv")
    return {(row['receptor'], row['train_type']): row['p95_similarity'] for _, row in df.iterrows()}

threshold_dict = load_threshold()

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

def load_dl_model_dynamic(model_name, target, mode):
    """加载受体对应的超参数和权重"""
    json_path = os.path.join(HYPERPARAMS_DIR, f"{target}_best_hyperparameters.json")
    weight_path = os.path.join(DL_WEIGHTS_BASE, target, mode, "ros", f"{model_name}.pth")
    
    with open(json_path, 'r') as f:
        params = json.load(f)[f"{mode}_ros_{model_name}"]
    
    model_map = {"GIN": GINModel, "GCN": GCNModel, "AFP": AttentiveFP, "GT": GraphTransformerModel}
    model_class = model_map[model_name]
    
    if model_name == "AFP":
        model = model_class(in_channels=32, hidden_channels=params['hidden_channels'], out_channels=1, edge_dim=11, num_layers=params['num_layers'], num_timesteps=params['num_timesteps'])
    elif model_name == "GT":
        model = model_class(in_channels=32, hidden_channels=params['hidden_channels'], out_channels=1, edge_dim=11, num_layers=params['num_layers'], dropout=params['dropout'], n_heads=params.get('n_heads', 4))
    else:
        model = model_class(in_channels=32, hidden_channels=params['hidden_channels'], out_channels=1, edge_dim=11, num_layers=params['num_layers'], dropout=params['dropout'])
    
    model.load_state_dict(torch.load(weight_path, map_location=device))
    return model.to(device).eval()

def run_prediction(target, mode, smiles_list):
    """运行预测"""
    all_probs = []
    smiles_list = clean_smiles_list(smiles_list)
    
    if (target, mode) in ENSEMBLE_TASKS:
        config = ENSEMBLE_SUBMODELS.get((target, mode))
        if not config:
            st.error(f"Ensemble configuration missing for {target}-{mode}")
            return None, None

        #ensemble ML
        for sampling, filename in config['ml']:
            path = os.path.join(BASE_PATH, target, "ml_final_models", mode, sampling, filename)
            if os.path.exists(path):
                model = load(path)
                tag = filename.split('+')[1].split('.')[0]
                X = calculate_features(smiles_list, tag)
                all_probs.append(model.predict_proba(X)[:, 1])

        #ensemble DL
        graph_data = [mol_to_graph_data_obj_simple(Chem.MolFromSmiles(s)) for s in smiles_list]
        loader = DataLoader(graph_data, batch_size=len(smiles_list))
        for m_name in config['dl']:
            try:
                model = load_dl_model_dynamic(m_name, target, mode)
                with torch.no_grad():
                    for data in loader:
                        data = data.to(device)
                        out = model(data.x.float(), data.edge_index.long(), data.edge_attr.float(), data.batch.long())
                        all_probs.append(torch.sigmoid(out).cpu().numpy().flatten())
            except Exception as e:
                st.warning(f"DL Model {m_name} failed: {e}")
                
        final_probs = np.mean(all_probs, axis=0)
    else:
        #single model
        model_file = SINGLE_MODEL_CONFIG.get((target, mode))
        path = os.path.join(BASE_PATH, target, "ml_final_models", mode, "none", model_file)
        tag = model_file.split('+')[1].split('.')[0]
        X = calculate_features(smiles_list, tag)
        model = load(path)
        final_probs = model.predict_proba(X)[:, 1]

    preds = (final_probs >= 0.5).astype(int)
    return preds, final_probs
    
def split_none(df):
    """划分数据"""
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=6, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=6, stratify=y_temp
    )
    return X_train, X_val, X_test

def array_to_fp(row):
    """将0/1数组转为RDKit指纹"""
    bv = ExplicitBitVect(len(row))
    for i, b in enumerate(row):
        if int(b) == 1:
            bv.SetBit(i)
    return bv

def smiles_to_fp(smiles_list):
    """SMILES转Morgan指纹"""
    fps = []
    valid_idx = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fps.append(fp)
            valid_idx.append(i)
    return fps, valid_idx

def calculate_ad(test_fps, train_fps, threshold, k=5):
    """基于top-k平均相似度的AD判定"""
    results = {}
    for idx, fp in enumerate(test_fps):
        sims = BulkTanimotoSimilarity(fp, train_fps)
        sims = np.array(sims)
        top_k = np.partition(sims, -k)[-k:]
        avg_sim = np.mean(top_k)
        results[idx] = 'Inside AD' if avg_sim >= threshold else 'Outside AD'
    return results

def load_train_fps(receptor, train_type):
    """加载训练集分子指纹"""
    key = (receptor, train_type)

    if key in AD_CACHE:
        return AD_CACHE[key]

    path = os.path.join("datasets", receptor, train_type, "morgan", "morgan.csv")
    df = pd.read_csv(path)

    X_train, _, _ = split_none(df)
    fps = [array_to_fp(row.values) for _, row in X_train.iterrows()]

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

#Streamlit
def main():
    st.title("Nuclear Receptor Activity Prediction Platform")
    st.image("Schematic diagram.png", caption="Schematic Diagram", use_column_width=True)
    
    # 所有任务
    ALL_TASKS = list(set(list(SINGLE_MODEL_CONFIG.keys()) + ENSEMBLE_TASKS))
    ALL_TASKS = sorted(ALL_TASKS)

    # 侧边栏
    st.sidebar.header("Target Configuration")
    all_targets = sorted(list(set([k[0] for k in SINGLE_MODEL_CONFIG.keys()] + [k[0] for k in ENSEMBLE_TASKS])))
    selected_target = st.sidebar.selectbox("Select Receptor", all_targets)
    
    available_modes = sorted(list(set(
        [k[1] for k in SINGLE_MODEL_CONFIG.keys() if k[0] == selected_target] + 
        [k[1] for k in ENSEMBLE_TASKS if k[0] == selected_target]
    )))
    selected_mode = st.sidebar.selectbox("Select Mode", available_modes)

    # 输入区域
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

    # 按钮
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

    with btn_col2:
        start_single = st.button("Start Calculation", use_container_width=True)
        start_all = st.button("Run All Targets", use_container_width=True)

    # 单任务预测
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

    # 全部任务预测
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

                    # 清理GPU缓存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    st.warning(f"{target}-{mode} failed: {e}")

                progress.progress((i + 1) / total)

            res_df = pd.DataFrame(all_results)

            # 原始结果
            st.subheader("All Predictions")
            st.dataframe(res_df)

            # 下载
            csv = res_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "all_predictions.csv", "text/csv")

if __name__ == "__main__":
    main()