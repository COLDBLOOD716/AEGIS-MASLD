import os
import glob
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

STATIC_DIM    = 1024
DYN_FEAT_DIM  = 10
EMBED_DIM     = 256
NUM_HEADS     = 4
TRANS_LAYERS  = 1
LSTM_LAYERS   = 1
SEQ_LEN       = 15
BATCH_SIZE    = 32
LR            = 1e-3
EPOCHS        = 120

KFOLD       = 5
DEVICE      = torch.device("cuda")
SAVE_DIR    = r"/path/to/save/models"
os.makedirs(SAVE_DIR, exist_ok=True)

DISEASES = ["CAD", "Stroke", "PAD", "CKD", "Cirrhosis", "HCC",
            "T2DM.Insulin.Dependent", "Hypothyroidism", "PCOS",
            "Heart.Failure", "Arrhythmias", "Extrahepatic.tumors", "Death"]

class LiverDataset(Dataset):
    def __init__(self, static_dir, dyn_dir, disease_name):
        self.dyn_files = sorted(glob.glob(os.path.join(dyn_dir, "*.csv")))
        self.static_dir = static_dir
        self.disease_name = disease_name
        self.pids = [os.path.splitext(os.path.basename(f))[0] for f in self.dyn_files]

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        pid = self.pids[idx]
        static_path = os.path.join(self.static_dir, f"{pid}-StaticMultimodal.pt")
        static_feat = torch.load(static_path)
        if not torch.is_tensor(static_feat):
            if isinstance(static_feat, dict) and 'feat' in static_feat:
                static_feat = static_feat['feat']
            else:
                static_feat = torch.tensor(static_feat, dtype=torch.float32)

        dyn_path = self.dyn_files[idx]
        df = pd.read_csv(dyn_path, index_col=0)
        habits = torch.tensor(df.iloc[:, :DYN_FEAT_DIM].values[:SEQ_LEN], dtype=torch.float32)
        labels = df[self.disease_name].values[:SEQ_LEN]
        if len(labels) < SEQ_LEN:
            pad = np.zeros(SEQ_LEN - len(labels), dtype=labels.dtype)
            labels = np.concatenate([labels, pad], axis=0)
        labels = torch.tensor(labels, dtype=torch.float32)
        ever = (labels.max() > 0).float()
        return static_feat, habits, labels, ever

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        T = x.size(1)
        x = x + self.pe[:T, :].unsqueeze(0)
        return x

class TransLSTMMultiTask(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.static2h0 = nn.Linear(STATIC_DIM, EMBED_DIM * LSTM_LAYERS)
        self.habits_proj = nn.Linear(DYN_FEAT_DIM, EMBED_DIM)
        enc_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS,
                                               dim_feedforward=EMBED_DIM*2, batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=TRANS_LAYERS)
        self.pos_enc = PositionalEncoding(EMBED_DIM, max_len=SEQ_LEN)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=EMBED_DIM, hidden_size=EMBED_DIM,
                            num_layers=LSTM_LAYERS, batch_first=True)
        self.attn_vector = nn.Parameter(torch.randn(EMBED_DIM))
        self.time_head = nn.Linear(EMBED_DIM, 1)
        self.event_head = nn.Linear(EMBED_DIM, 1)

    def forward(self, static_feat, habits):
        B, T, _ = habits.shape
        h0 = self.static2h0(static_feat).view(LSTM_LAYERS, B, EMBED_DIM)
        c0 = torch.zeros_like(h0, device=habits.device)

        x = self.habits_proj(habits)
        x = self.pos_enc(x)
        x = self.dropout(x)
        x = self.transformer(x)
        out_seq, (h_n, c_n) = self.lstm(x, (h0, c0))
        time_logits = self.time_head(out_seq).squeeze(-1)
        attn_scores = torch.matmul(out_seq, self.attn_vector)
        attn_w = torch.softmax(attn_scores, dim=1)
        event_rep = (out_seq * attn_w.unsqueeze(-1)).sum(dim=1)
        event_logits = self.event_head(event_rep).squeeze(-1)
        return time_logits, event_logits

def compute_auc(preds, targets):
    p = preds.detach().cpu().numpy().ravel()
    t = targets.detach().cpu().numpy().ravel()
    if np.unique(t).size < 2:
        return 0.5
    return roc_auc_score(t, p)

def train_and_save(disease_name):
    dataset = LiverDataset(static_dir="/path/to/static",
                           dyn_dir="/path/to/dynamic",
                           disease_name=disease_name)

    total_patients = len(dataset)
    pos_event = 0
    total_time_pos = 0
    total_time_all = 0
    for dyn_f in dataset.dyn_files:
        df = pd.read_csv(dyn_f, index_col=0)
        labels = df[disease_name].values[:SEQ_LEN]
        if labels.max() > 0:
            pos_event += 1
        total_time_pos += labels.sum()
        total_time_all += SEQ_LEN

    pos_weight_event = float((total_patients - pos_event) / max(1, pos_event))
    pos_weight_time = float((total_time_all - total_time_pos) / max(1, total_time_pos))

    kf = KFold(n_splits=KFOLD, shuffle=True, random_state=42)

    best_weighted_yearly_auc = 0.0
    best_path_weighted_yearly = None
    metrics_summary = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        train_ds = Subset(dataset, train_idx)
        val_ds   = Subset(dataset, val_idx)
        train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = TransLSTMMultiTask().to(DEVICE)
        bce_time  = nn.BCEWithLogitsLoss(reduction='none')
        bce_event = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight_event, device=DEVICE))
        optimizer = optim.Adam(model.parameters(), lr=LR)

        for epoch in range(1, EPOCHS+1):
            model.train()
            total_loss = 0.0
            for static_feat, habits, seq_y, ever_y in train_dl:
                static_feat = static_feat.to(DEVICE).float()
                habits = habits.to(DEVICE).float()
                seq_y = seq_y.to(DEVICE).float()
                ever_y = ever_y.to(DEVICE).float()

                t_logits, e_logits = model(static_feat, habits)
                B, T = seq_y.shape
                time_idx = torch.arange(T, device=seq_y.device).unsqueeze(0).repeat(B, 1)
                event_pos_matrix = torch.where(seq_y.bool(), time_idx, torch.full_like(time_idx, T))
                first_idx = event_pos_matrix.min(dim=1).values
                mask = (time_idx <= first_idx.unsqueeze(1)).float()
                loss_time_all = bce_time(t_logits, seq_y)
                weight_matrix = 1.0 + (pos_weight_time - 1.0) * seq_y
                loss_time_all = loss_time_all * weight_matrix
                masked_sum = (loss_time_all * mask).sum()
                denom = mask.sum().clamp(min=1.0)
                loss_time = masked_sum / denom
                loss_event = bce_event(e_logits, ever_y).mean()
                loss = loss_time + loss_event

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            model.eval()
            t_probs_list, t_targets_list = [], []
            ev_probs_list, ev_targets_list = [], []
            with torch.no_grad():
                for sf, hb, seq_y, ev in val_dl:
                    sf, hb, seq_y, ev = sf.to(DEVICE).float(), hb.to(DEVICE).float(), seq_y.to(DEVICE).float(), ev.to(DEVICE).float()
                    t_logits, e_logits = model(sf, hb)
                    t_probs_list.append(torch.sigmoid(t_logits).cpu())
                    t_targets_list.append(seq_y.cpu())
                    ev_probs_list.append(torch.sigmoid(e_logits).cpu())
                    ev_targets_list.append(ev.cpu())

            t_probs_all = torch.cat(t_probs_list).numpy()
            t_targets_all = torch.cat(t_targets_list).numpy()
            true_first = np.argmax(t_targets_all, axis=1) + 1
            true_first[np.max(t_targets_all, axis=1) == 0] = 0
            event_mask = (true_first > 0)
            num_events = int(event_mask.sum())
            weighted_mean_auc = np.nan
            if num_events > 0:
                counts = np.bincount(true_first[event_mask], minlength=SEQ_LEN+1)[1:]
                w = counts.astype(float) / num_events
                auc_per_t = np.full(SEQ_LEN, np.nan)
                for t in range(1, SEQ_LEN+1):
                    risk_mask = (true_first == 0) | (true_first >= t)
                    if not np.any(risk_mask):
                        continue
                    y_true_incident = (true_first == t).astype(int)
                    y_true = y_true_incident[risk_mask]
                    y_score = t_probs_all[risk_mask, t-1]
                    pos = int(y_true.sum())
                    neg = int(y_true.shape[0] - pos)
                    if pos > 0 and neg > 0:
                        try:
                            auc_per_t[t-1] = roc_auc_score(y_true, y_score)
                        except Exception:
                            auc_per_t[t-1] = np.nan
                valid = (w > 0) & (~np.isnan(auc_per_t))
                if valid.any():
                    weighted_mean_auc = np.sum(w[valid] * auc_per_t[valid]) / np.sum(w[valid])

            if not np.isnan(weighted_mean_auc) and weighted_mean_auc > best_weighted_yearly_auc:
                best_weighted_yearly_auc = float(weighted_mean_auc)
                best_path_weighted_yearly = os.path.join(SAVE_DIR, f"{disease_name}_weighted_yearly_best_fold{fold+1}_ep{epoch}.pth")
                torch.save(model.state_dict(), best_path_weighted_yearly)
                f1_weighted = f1_score((true_first > 0).astype(int), t_probs_all.max(axis=1) > 0.5)
                prec_weighted = precision_score((true_first > 0).astype(int), t_probs_all.max(axis=1) > 0.5, zero_division=0)
                rec_weighted = recall_score((true_first > 0).astype(int), t_probs_all.max(axis=1) > 0.5, zero_division=0)
                metrics_summary.append({
                    "disease": disease_name,
                    "model_type": "weighted_yearly",
                    "auc": float(weighted_mean_auc),
                    "f1": float(f1_weighted),
                    "precision": float(prec_weighted),
                    "recall": float(rec_weighted),
                    "fold": int(fold+1),
                    "epoch": int(epoch),
                    "model_path": best_path_weighted_yearly
                })

            print(f"Epoch {epoch:02d} | TrainLoss: {total_loss:.4f} | Weighted_yearly_AUC: {weighted_mean_auc:.4f}")

    if len(metrics_summary) > 0:
        metrics_df = pd.DataFrame(metrics_summary)
        metrics_csv_path = os.path.join(SAVE_DIR, f"{disease_name}_metrics_summary.csv")
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Saved metrics summary: {metrics_csv_path}")

    return {
        "weighted_yearly_path": best_path_weighted_yearly,
        "best_weighted_yearly_auc": best_weighted_yearly_auc
    }

if __name__ == "__main__":
    for disease in DISEASES:
        res = train_and_save(disease)
        print(f"Saved model for disease {disease}: {res}")
