import streamlit as st
st.set_page_config(page_title="NURA Activity Predictor", layout="wide")

import os, json, warnings, numpy as np, pandas as pd, torch
from io import BytesIO
from joblib import load
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, MACCSkeys, Draw
from torch_geometric.loader import DataLoader
import deepchem as dc
from streamlit_ketcher import st_ketcher

from utils import mol_to_graph_data_obj_simple
from model import GINModel, GCNModel, GraphTransformerModel, GATModel
from torch_geometric.nn.models import AttentiveFP

warnings.filterwarnings("ignore")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PRETRAIN_MOL2VEC_PATH = './model_300dim.pkl'
HYPERPARAMS_DIR = './best_hyperparameters'
ML_BASE = './NURA_Baseline_ml'
GNN_BASE = './NURA_Baseline_gnn'
CONSENSUS_BASE = './NURA_consensus'
ML_TOP_K, GNN_TOP_K, N_REPS = 5, 2, 5

GNN_CLASS_MAP = {
    "GT": GraphTransformerModel, "GIN": GINModel,
    "GCN": GCNModel, "GAT": GATModel, "AFP": AttentiveFP,
}

CONSENSUS_CONFIGS = {
    ('FXR','binder'):('none','none'), ('PPARD','binder'):('none','none'),
    ('RXR','binder'):('none','none'), ('ERB','binder'):('none','none'),
    ('PR','binder'):('none','none'), ('AR','binder'):('none','none'),
    ('ERA','binder'):('none','none'), ('PXR','binder'):('none','none'),
    ('PPARG','binder'):('none','none'), ('GR','binder'):('none','none'),
    ('ERA','antagonist'):('none','none'), ('GR','antagonist'):('none','none'),
    ('AR','antagonist'):('none','none'), ('PR','antagonist'):('none','none'),
    ('PPARG','antagonist'):('rus','ros'), ('FXR','antagonist'):('rus','ros'),
    ('ERB','antagonist'):('rus','ros'),
    ('AR','agonist'):('none','none'), ('PPARD','agonist'):('none','none'),
    ('GR','agonist'):('none','none'), ('ERA','agonist'):('none','none'),
    ('PPARG','agonist'):('none','none'), ('PXR','agonist'):('none','none'),
    ('RXR','agonist'):('rus','ros'), ('ERB','agonist'):('rus','ros'),
    ('PR','agonist'):('rus','ros'), ('FXR','agonist'):('rus','ros'),
    ('TSHR', 'agonist'): ('none', 'none'),
    ('TSHR', 'antagonist'): ('rus', 'ros'),
    ('TR', 'antagonist'): ('none', 'none'),
    ('NIS', 'binder'): ('rus', 'ros'),
    ('TPO', 'binder'): ('none', 'none'),
    ('DIO1', 'binder'): ('rus', 'ros'),
    ('DIO2', 'binder'): ('none', 'none'),
    ('DIO3', 'binder'): ('none', 'none')
}

AD_CACHE = {}
_MODEL_CACHE = {}


def get_top_models(target, mode, sampling, model_type='ml', k=5, metric='MCC'):
    ck = (model_type, target, mode, sampling, k, metric)
    if ck in _MODEL_CACHE: return _MODEL_CACHE[ck]

    base = os.path.join(ML_BASE if model_type == 'ml' else GNN_BASE, target, mode, sampling)
    if model_type == 'ml':
        m_p = os.path.join(base, 'results_10to1', 'results_mean.xlsx')
        s_p = os.path.join(base, 'results_10to1', 'results_std.xlsx')
    else:
        d = os.path.join(base, 'graph_results', 'results_mean_std')
        m_p = os.path.join(d, 'results_mean.xlsx'); s_p = os.path.join(d, 'results_std.xlsx')

    dm = pd.read_excel(m_p, sheet_name='test' if model_type == 'ml' else 'Test')
    ds = pd.read_excel(s_p, sheet_name='test' if model_type == 'ml' else 'Test')

    if model_type == 'ml':
        df = pd.DataFrame({'model': dm['model'], 'rep': dm['rep'], 'm': dm[metric], 's': ds[metric]})
    else:
        df = pd.DataFrame({'model': dm['Model'], 'm': dm[metric], 's': ds[metric]})
    df['lb'] = df['m'] - df['s']
    top = df.sort_values('lb', ascending=False).head(k)
    result = list(zip(top['model'], top['rep'])) if model_type == 'ml' else top['model'].tolist()
    _MODEL_CACHE[ck] = result
    return result


def calc_features(smiles_list, tag):
    tag = tag.lower(); mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    if 'desc' in tag:
        return np.array([[d(m) for _, d in Descriptors.descList] for m in mols])
    if 'maccs' in tag:
        return np.array([list(MACCSkeys.GenMACCSKeys(m)) for m in mols])
    if 'rdk' in tag:
        return np.array([list(Chem.RDKFingerprint(m)) for m in mols])
    if 'morgan' in tag:
        return np.array([list(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)) for m in mols])
    if 'mol2vec' in tag:
        fz = dc.feat.Mol2VecFingerprint(pretrain_model_path=PRETRAIN_MOL2VEC_PATH)
        return np.array([fz.featurize(s)[0].tolist() for s in smiles_list])


def load_gnn_model(model_name, target, mode, gnn_sampling, rep):
    params = json.load(open(os.path.join(HYPERPARAMS_DIR, f'{target}_best_hyperparameters.json')))
    p = params[f'{mode}_{gnn_sampling}_{model_name}']
    pth = os.path.join(GNN_BASE, target, mode, gnn_sampling, 'graph_models', f'Replicate_{rep}', f'{model_name}.pth')
    cls = GNN_CLASS_MAP[model_name]
    kw = dict(in_channels=32, hidden_channels=p['hidden_channels'], out_channels=1,
              edge_dim=11, num_layers=p['num_layers'], dropout=p['dropout'])
    if cls == GraphTransformerModel: kw['n_heads'] = p['n_heads']
    elif cls == AttentiveFP: kw['num_timesteps'] = p['num_timesteps']
    m = cls(**kw)
    m.load_state_dict(torch.load(pth, map_location=device)); m.to(device); m.eval()
    return m


def run_prediction(target, mode, smiles_list):
    """Stacking 预测: 5个副本基模型→元模型→均值"""
    config = CONSENSUS_CONFIGS.get((target, mode))
    if not config: st.error(f"No consensus config for {target}-{mode}"); return None, None
    ml_s, gnn_s = config

    # 标准化 SMILES
    clean = []
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        clean.append(Chem.MolToSmiles(m, isomericSmiles=False) if m else s)

    ml_ens = get_top_models(target, mode, ml_s, 'ml', ML_TOP_K)
    gnn_models = get_top_models(target, mode, gnn_s, 'gnn', GNN_TOP_K)
    consensus_dir = os.path.join(CONSENSUS_BASE, target, mode, f'consensus_stacking_{ml_s}_{gnn_s}')

    ft_cache = {ft: calc_features(clean, ft) for ft in set(f for _, f in ml_ens)}
    graphs = [mol_to_graph_data_obj_simple(Chem.MolFromSmiles(s)) for s in clean]
    loader = DataLoader(graphs, batch_size=len(clean))

    rep_probs = []
    for rep in range(1, N_REPS + 1):
        meta_path = os.path.join(consensus_dir, f'stacking_model_{rep}.joblib')
        if not os.path.exists(meta_path):
            st.warning(f"Missing meta-model: replicate {rep}"); continue

        all_probs = []
        for name, ft in ml_ens:
            mp = os.path.join(ML_BASE, target, mode, ml_s, 'final_models', f'Replicate_{rep}', f'{name}_{ft}.joblib')
            if os.path.exists(mp) and ft in ft_cache:
                all_probs.append(load(mp).predict_proba(ft_cache[ft])[:, 1])

        for name in gnn_models:
            try:
                m = load_gnn_model(name, target, mode, gnn_s, rep)
                with torch.no_grad():
                    for d in loader:
                        d = d.to(device)
                        out = m(d.x.float(), d.edge_index.long(), d.edge_attr.float(), d.batch.long())
                        all_probs.append(torch.sigmoid(out).cpu().numpy().flatten())
            except Exception: continue

        if not all_probs: continue
        rep_probs.append(load(meta_path).predict_proba(np.column_stack(all_probs))[:, 1])

    if not rep_probs:
        st.error(f"No valid replicates for {target}-{mode}"); return None, None

    final = np.mean(rep_probs, axis=0)
    return (final >= 0.5).astype(int), final


# ── Applicability Domain ──
@st.cache_data
def load_threshold():
    df = pd.read_csv("similarity_threshold.csv")
    return {(r['receptor'], r['train_type']): r['p95_similarity'] for _, r in df.iterrows()}

threshold_dict = load_threshold()


def smiles_to_fp(smiles_list, nBits=2048):
    fps, valid = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol:
            arr = np.zeros(nBits, dtype=np.int8)
            for bit in AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nBits).GetOnBits():
                arr[bit] = 1
            fps.append(arr); valid.append(i)
    return np.array(fps), valid


def calculate_ad(test_fps, train_fps, threshold, k=5):
    inter = test_fps @ train_fps.T
    union = test_fps.sum(1)[:, None] + train_fps.sum(1)[None, :] - inter
    sims = np.nan_to_num(inter / union, nan=0.0)
    results = {}
    for i in range(test_fps.shape[0]):
        row = sims[i]
        top_k = np.partition(row, -k)[-k:] if len(row) >= k else row
        results[i] = 'Inside AD' if top_k.mean() >= threshold else 'Outside AD'
    return results


def load_train_fps(receptor, train_type):
    key = (receptor, train_type)
    if key in AD_CACHE: return AD_CACHE[key]
    path = os.path.join("datasets", receptor, train_type, "morgan", "morgan.csv")
    fps = pd.read_csv(path).iloc[:, 1:-1].values.astype(np.int8)
    AD_CACHE[key] = fps
    return fps


def run_ad(smiles_list, receptor, mode):
    train_type = f"{mode}_train"
    threshold = threshold_dict.get((receptor, train_type))
    if threshold is None: return ["-"] * len(smiles_list)

    test_fps, valid_idx = smiles_to_fp(smiles_list)
    train_fps = load_train_fps(receptor, train_type)
    ad_raw = calculate_ad(test_fps, train_fps, threshold)
    idx_map = {orig: i for i, orig in enumerate(valid_idx)}

    results = []
    for i in range(len(smiles_list)):
        results.append(ad_raw[idx_map[i]] if i in idx_map else "Invalid SMILES")
    return results


# ── Streamlit UI ──
def main():
    st.title("Nuclear Receptor Activity Prediction Platform")
    st.image("Schematic diagram.png", caption="Schematic Diagram", use_column_width=True)

    ALL_TASKS = sorted(CONSENSUS_CONFIGS.keys())

    st.sidebar.header("Target Configuration")
    all_targets = sorted(set(k[0] for k in ALL_TASKS))
    selected_target = st.sidebar.selectbox("Select Receptor", all_targets)
    available_modes = sorted(set(k[1] for k in ALL_TASKS if k[0] == selected_target))
    selected_mode = st.sidebar.selectbox("Select Mode", available_modes)

    # ── Tutorial ──
    with st.sidebar.expander("📖 Tutorial"):
        st.markdown("""欢迎来到核受体活性预测平台！

**1. 平台概述**

该平台预测小分子在不同效应（结合剂、激动剂、拮抗剂）下对各种靶标的活性。

**2. 使用说明**

**第一步：选择你的靶标。** 先选择你想要预测的核受体，然后选择效应类型：结合剂、激动剂或拮抗剂。

**第二步：输入你的分子。** 有三种输入可供选择，选择一种输入方法：
- **Draw Molecule：** 使用化学结构编辑器绘制分子，绘制结束后需点击Apply，编辑器下方会显示SMILES。
- **SMILES String：** 直接粘贴分子的SMILES。
- **Batch Excel Upload：** 上传带有SMILES列的Excel文件批量预测（如有其他列，也会被保留）。

**第三步：开始你的预测。**
- **Start Calculation：** 仅预测所选的靶标和效应类型。
- **Run all Targets：** 预测所有的靶标和效应类型。

**第四步：解读输出结果。**

输出结果包括SMILES、Probability、Outcome、Applicability Domain，分别表示输入分子、预测概率、结果和适用域。预测概率在0-1之间，值越接近1，表明更可能表现出效应。概率≥0.5，结果输出为Active，概率<0.5，结果输出为Inactive。适用域包括Inside AD和Outside AD，Inside AD表明分子与训练数据较相似，预测结果可靠，Outside AD则相反。

**第五步：下载输出结果。**

在"Run all Targets"之后，您可以使用底部的"Download Results"按钮将所有预测下载为Excel文件。""")

    input_type = st.radio("Input Method", ["Draw Molecule", "SMILES String", "Batch Excel Upload"])
    smiles_list = []
    original_df = None  # 保存上传的原始 DataFrame，用于结果合并

    if input_type == "Draw Molecule":
        drawn = st_ketcher("")
        if drawn:
            st.write(f"Generated SMILES: {drawn}")
            smiles_list = [drawn]
    elif input_type == "SMILES String":
        s = st.text_input("Enter SMILES")
        if s: smiles_list = [s.strip()]
    else:
        file = st.file_uploader("Upload Excel file (must contain a 'SMILES' column)", type=["xlsx"])
        # Example Excel download
        with open("example.xlsx", "rb") as f:
            st.download_button(
                "📥 Download Example File", f.read(),
                file_name="example.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                help="Download an example Excel file to see the required format."
            )
        if file is not None:
            original_df = pd.read_excel(file, engine='openpyxl')
            if "SMILES" in original_df.columns:
                smiles_list = original_df["SMILES"].dropna().tolist()

    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
    with btn_col2:
        start_single = st.button("Start Calculation", use_container_width=True)
        start_all = st.button("Run All Targets", use_container_width=True)

    # Single task
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
                    "Applicability Domain": ad_results,
                })
                # 如果上传了 CSV，保留原始列
                if original_df is not None:
                    res_df = original_df.merge(res_df, on="SMILES", how="left")
                st.dataframe(res_df)
                if len(smiles_list) == 1:
                    mol = Chem.MolFromSmiles(smiles_list[0])
                    if mol: st.image(Draw.MolToImage(mol, size=(300, 300)))

    # Run all
    if start_all and smiles_list:
        st.warning("Running all targets may take several minutes.")
        with st.spinner("Running all targets..."):
            all_results = []
            progress = st.progress(0)
            for i, (target, mode) in enumerate(ALL_TASKS):
                try:
                    preds, probs = run_prediction(target, mode, smiles_list)
                    ad_results = run_ad(smiles_list, target, mode)
                    if preds is None: continue
                    for j, smi in enumerate(smiles_list):
                        all_results.append({
                            "SMILES": smi, "Target": target, "Mode": mode,
                            "Probability": probs[j],
                            "Outcome": "Active" if preds[j] == 1 else "Inactive",
                            "Applicability Domain": ad_results[j],
                        })
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                except Exception as e:
                    st.warning(f"{target}-{mode} failed: {e}")
                progress.progress((i + 1) / len(ALL_TASKS))

            res_df = pd.DataFrame(all_results)
            # 如果上传了 CSV，保留原始列
            if original_df is not None:
                res_df = original_df.merge(res_df, on="SMILES", how="left")
            st.subheader("All Predictions")
            st.dataframe(res_df)
            # 将结果写入 xlsx 避免 CSV 中文乱码
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                res_df.to_excel(writer, index=False, sheet_name='Predictions')
            output.seek(0)
            st.download_button(
                "Download Results", output,
                "all_predictions.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


if __name__ == "__main__":
    main()
