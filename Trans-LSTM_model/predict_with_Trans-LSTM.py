import os
import glob
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

STATIC_DIM   = 1024
DYN_FEAT_DIM = 10
EMBED_DIM    = 256
NUM_HEADS    = 4
TRANS_LAYERS = 1
LSTM_LAYERS  = 1
SEQ_LEN      = 15
DEVICE       = torch.device("cuda")

NEW_STATIC_DIR = r'PATH_TO_STATIC_FEATURES'
DYN_DIR        = r'PATH_TO_DYN_FEATURES'
MODEL_DIR      = r'PATH_TO_MODELS'
OUTPUT_DIR     = r'PATH_TO_OUTPUT'
os.makedirs(OUTPUT_DIR, exist_ok=True)

EXTERNAL_LABELS_CSV = r'PATH_TO_EXTERNAL_LABELS'

DISEASES = [
    "T2DM.Insulin.Dependent", "Cirrhosis", "CAD", "Stroke", "Heart.Failure",
    "Arrhythmias", "PAD", "CKD", "Extrahepatic.tumors", "Hypothyroidism", "PCOS", "HCC"
]

class InferenceDataset(Dataset):
    def __init__(self, static_dir, dyn_dir):
        self.static_files = sorted(glob.glob(os.path.join(static_dir, "*-StaticMultimodal.pt")))
        self.dyn_dir = dyn_dir

    def __len__(self):
        return len(self.static_files)

    def __getitem__(self, idx):
        static_path = self.static_files[idx]
        pid = os.path.basename(static_path).replace("-StaticMultimodal.pt", "")
        static_feat = torch.load(static_path)
        if isinstance(static_feat, torch.Tensor):
            if static_feat.ndim == 2 and static_feat.shape[0] == 1:
                static_feat = static_feat.squeeze(0)
            elif static_feat.ndim > 2:
                static_feat = static_feat.view(-1)
        else:
            static_feat = torch.tensor(static_feat, dtype=torch.float32)

        csv_path = os.path.join(self.dyn_dir, f"{pid}.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Time series file not found: {csv_path}")
        df = pd.read_csv(csv_path, index_col=0)

        arr = df.iloc[:SEQ_LEN, :DYN_FEAT_DIM].values.astype(np.float32)
        if arr.shape[0] < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - arr.shape[0], DYN_FEAT_DIM), dtype=np.float32)
            arr = np.vstack([arr, pad])
        habits = torch.from_numpy(arr)

        return pid, static_feat, habits

class TransLSTMMultiTask(nn.Module):
    def __init__(self):
        super().__init__()
        self.static2h0 = nn.Linear(STATIC_DIM, EMBED_DIM * LSTM_LAYERS)
        self.habits_proj = nn.Linear(DYN_FEAT_DIM, EMBED_DIM)
        enc_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS,
                                               dim_feedforward=EMBED_DIM*2, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=TRANS_LAYERS)
        self.lstm = nn.LSTM(input_size=EMBED_DIM, hidden_size=EMBED_DIM,
                            num_layers=LSTM_LAYERS, batch_first=True)
        self.time_head = nn.Linear(EMBED_DIM, 1)
        self.event_head = nn.Linear(EMBED_DIM, 1)

    def forward(self, static_feat, habits):
        if static_feat.dim() == 3 and static_feat.size(1) == 1:
            static_feat = static_feat.squeeze(1)
        if static_feat.dim() == 1:
            static_feat = static_feat.unsqueeze(0)

        if habits.dim() == 2:
            habits = habits.unsqueeze(0)
        if habits.dim() == 4 and habits.size(1) == 1:
            habits = habits.squeeze(1)

        B, T, _ = habits.shape
        h0 = self.static2h0(static_feat).view(LSTM_LAYERS, B, EMBED_DIM)
        c0 = torch.zeros_like(h0, device=h0.device)
        x = self.habits_proj(habits)
        x = self.transformer(x)
        out_seq, (h_n, _) = self.lstm(x, (h0, c0))
        time_logits = self.time_head(out_seq).squeeze(-1)
        event_logits = self.event_head(h_n[-1]).squeeze(-1)
        return time_logits, event_logits

def inference_all_diseases():
    ds = InferenceDataset(NEW_STATIC_DIR, DYN_DIR)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    disease_models = {}
    for disease in DISEASES:
        model_path = os.path.join(MODEL_DIR, f"{disease}.pth")
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}, skip")
            continue
        model = TransLSTMMultiTask().to(DEVICE)
        state = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()
        disease_models[disease] = model
        print(f"Loaded model: {disease}")

    if len(disease_models) == 0:
        print("No disease models loaded, exiting.")
        return

    disease_pos_counts = {d: 0 for d in disease_models.keys()}
    total_patients = 0
    per_patient_rows = []

    with torch.no_grad():
        for pid, static_feat, habits in dl:
            pid_val = pid[0] if isinstance(pid, (list, tuple)) else pid
            pid_str = pid_val if isinstance(pid_val, str) else str(pid_val)
            total_patients += 1
            static_feat = static_feat.to(DEVICE).float()
            habits = habits.to(DEVICE).float()

            pred_dict = {"Year": [f"Year{i+1}" for i in range(SEQ_LEN)]}
            per_row = {"ID": pid_str}

            for disease, model in disease_models.items():
                time_logits, _ = model(static_feat, habits)
                probs = torch.sigmoid(time_logits).cpu().numpy().reshape(-1)[:SEQ_LEN]
                pred_dict[disease] = np.round(probs.tolist(), 6)

                is_positive = bool((probs > 0.5).any())
                maxprob = float(probs.max())
                per_row[f"{disease}_pred15_binary"] = int(is_positive)
                per_row[f"{disease}_pred15_maxprob"] = maxprob

                if is_positive:
                    disease_pos_counts[disease] += 1

            out_path = os.path.join(OUTPUT_DIR, f"{pid_str}_pred.csv")
            pd.DataFrame(pred_dict).to_csv(out_path, index=False)
            print(f"Saved prediction: {os.path.basename(out_path)}")
            per_patient_rows.append(per_row)

    rows = []
    print("\n===== 15-year prevalence summary (threshold 0.5) =====")
    for disease in sorted(disease_models.keys()):
        pos = disease_pos_counts[disease]
        rate = pos / max(1, total_patients)
        print(f"{disease:25s} : {pos}/{total_patients} = {rate*100:.2f}%")
        rows.append({"disease": disease, "positive_count": pos, "total": total_patients, "prevalence": rate})

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(OUTPUT_DIR, "summary_15yr_prevalence.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved 15-year prevalence summary: {summary_path}")

    preds_summary_df = pd.DataFrame(per_patient_rows)
    preds_summary_path = os.path.join(OUTPUT_DIR, "pred_15yr_summary.csv")
    preds_summary_df.to_csv(preds_summary_path, index=False)
    print(f"Saved per-patient 15-year summary: {preds_summary_path}")

    if not os.path.exists(EXTERNAL_LABELS_CSV):
        print(f"External labels file not found: {EXTERNAL_LABELS_CSV}, skip AUROC/AUPRC computation.")
        return

    try:
        ext_df = pd.read_csv(EXTERNAL_LABELS_CSV, encoding='utf-8-sig', engine='python')
    except Exception:
        ext_df = pd.read_csv(EXTERNAL_LABELS_CSV, engine='python')

    ext_df.columns = [c.strip().lstrip('\ufeff') for c in ext_df.columns]

    if 'ID' not in ext_df.columns:
        raise KeyError(f"'ID' column not found in external labels, current columns: {ext_df.columns.tolist()}")
    ext_df['ID'] = ext_df['ID'].astype(str)

    for d in DISEASES:
        if d in ext_df.columns:
            ext_df[d] = pd.to_numeric(ext_df[d], errors='coerce').fillna(0).astype(int)
        else:
            ext_df[d] = 0
            print(f"Warning: missing disease column '{d}', filled with 0.")

    merged = pd.merge(preds_summary_df, ext_df[['ID'] + DISEASES], on='ID', how='inner')
    n_overlap = len(merged)
    if n_overlap == 0:
        print("No overlapping samples between predictions and external labels, cannot compute AUROC/AUPRC.")
        return
    print(f"Found {n_overlap} overlapping samples, computing AUROC/AUPRC...")

    metric_rows = []
    for d in DISEASES:
        score_col = f"{d}_pred15_maxprob"
        true_col = d
        if score_col not in merged.columns:
            print(f"Missing prediction score column {score_col}, skip {d}")
            continue
        y_score = merged[score_col].astype(float).values
        y_true = merged[true_col].astype(int).values
        unique = np.unique(y_true)
        if unique.size < 2:
            auroc = np.nan
            auprc = np.nan
            print(f"{d:25s} skipped (single-class labels: {unique.tolist()})")
        else:
            try:
                auroc = roc_auc_score(y_true, y_score)
            except Exception:
                auroc = np.nan
            try:
                auprc = average_precision_score(y_true, y_score)
            except Exception:
                auprc = np.nan
            print(f"{d:25s}  AUROC: {auroc}  AUPRC: {auprc}")

        metric_rows.append({'disease': d, 'n_overlap': n_overlap, 'auroc': auroc, 'auprc': auprc})

    metrics_df = pd.DataFrame(metric_rows)
    final_summary = summary_df.merge(metrics_df, on='disease', how='left')
    final_summary_path = os.path.join(OUTPUT_DIR, "summary_15yr_prevalence_with_metrics.csv")
    final_summary.to_csv(final_summary_path, index=False)
    print(f"Saved summary with AUROC/AUPRC: {final_summary_path}")

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")
    inference_all_diseases()
