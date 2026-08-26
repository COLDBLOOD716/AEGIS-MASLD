from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold


STATIC_DIM = 1024
DYN_FEAT_DIM = 10
EMBED_DIM = 128
NUM_HEADS = 4
TRANS_LAYERS = 2
LSTM_LAYERS = 2
SEQ_LEN = 15
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 120
KFOLD = 5
DEFAULT_SEED = 42

LIFESTYLE_COLUMNS = [
    "High.fat.diet",
    "Sports",
    "Sugary.drinks",
    "Coffee.Consumption",
    "Inadequate.Fruit.and.Vegetable.Intake",
    "Sedentary.Office.Work",
    "Smoking",
    "Poor.Sleep",
    "Chronic.Stress",
    "Alcohol",
]

DISEASES = [
    "CAD",
    "Stroke",
    "PAD",
    "CKD",
    "Cirrhosis",
    "HCC",
    "T2DM.Insulin.Dependent",
    "Hypothyroidism",
    "PCOS",
    "Heart.Failure",
    "Arrhythmias",
    "Extrahepatic.tumors",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def diagnosis_prefix_mask(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim == 1:
        labels = labels.unsqueeze(0)
        squeeze = True
    elif labels.ndim == 2:
        squeeze = False
    else:
        raise ValueError("labels must have shape [T] or [B, T]")
    positive = labels > 0.5
    length = labels.size(1)
    indices = torch.arange(length, device=labels.device).unsqueeze(0)
    first = torch.where(positive, indices, length).min(dim=1).values
    valid = indices <= first.unsqueeze(1)
    return valid.squeeze(0) if squeeze else valid


class LiverDataset(Dataset):
    def __init__(
        self,
        static_dir: str | os.PathLike[str],
        dynamic_dir: str | os.PathLike[str],
        disease_name: str,
        patient_ids: Sequence[str] | None = None,
    ) -> None:
        self.static_dir = Path(static_dir)
        self.dynamic_dir = Path(dynamic_dir)
        self.disease_name = disease_name
        if disease_name not in DISEASES:
            raise ValueError(f"Unsupported disease: {disease_name}")
        if not self.static_dir.is_dir() or not self.dynamic_dir.is_dir():
            raise FileNotFoundError("static_dir and dynamic_dir must both exist")

        static_ids = {
            path.name[: -len("-StaticMultimodal.pt")]
            for path in self.static_dir.glob("*-StaticMultimodal.pt")
        }
        dynamic_ids = {path.stem for path in self.dynamic_dir.glob("*.csv")}
        if static_ids != dynamic_ids:
            static_only = sorted(static_ids - dynamic_ids)
            dynamic_only = sorted(dynamic_ids - static_ids)
            raise ValueError(
                "ID mismatch between static and dynamic inputs; "
                f"static_only={static_only[:10]}, dynamic_only={dynamic_only[:10]}"
            )
        if patient_ids is None:
            self.pids = sorted(static_ids)
        else:
            requested = [str(pid) for pid in patient_ids]
            missing = sorted(set(requested) - static_ids)
            if missing:
                raise ValueError(f"Requested IDs are missing: {missing[:10]}")
            self.pids = requested
        if not self.pids:
            raise ValueError("No matched patients found")

        self.static_features: list[torch.Tensor] = []
        self.habits: list[torch.Tensor] = []
        self.labels: list[torch.Tensor] = []
        self.valid_masks: list[torch.Tensor] = []
        self.ever_labels: list[float] = []
        for pid in self.pids:
            self._load_patient(pid)
        self.ever_labels = np.asarray(self.ever_labels, dtype=np.int64)

    def _load_patient(self, pid: str) -> None:
        static_path = self.static_dir / f"{pid}-StaticMultimodal.pt"
        raw_static = torch.load(static_path, map_location="cpu")
        if isinstance(raw_static, dict):
            if "feat" not in raw_static:
                raise ValueError(f"{static_path} is a dict without a 'feat' tensor")
            raw_static = raw_static["feat"]
        static = torch.as_tensor(raw_static, dtype=torch.float32).reshape(-1)
        if static.numel() != STATIC_DIM:
            raise ValueError(
                f"{static_path} contains {static.numel()} values; expected {STATIC_DIM}"
            )
        if not bool(torch.isfinite(static).all()):
            raise ValueError(f"Non-finite static feature in {static_path}")

        dynamic_path = self.dynamic_dir / f"{pid}.csv"
        frame = pd.read_csv(dynamic_path, index_col=0)
        required = LIFESTYLE_COLUMNS + [self.disease_name]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"{dynamic_path} is missing columns: {missing}")
        if len(frame) != SEQ_LEN:
            raise ValueError(f"{dynamic_path} has {len(frame)} rows; expected {SEQ_LEN}")

        habits_np = frame[LIFESTYLE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        labels_np = pd.to_numeric(frame[self.disease_name], errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(habits_np).all() or not np.isfinite(labels_np).all():
            raise ValueError(f"Non-finite dynamic value in {dynamic_path}")
        if not set(np.unique(habits_np)).issubset({0.0, 1.0}):
            raise ValueError(f"Lifestyle values must be binary in {dynamic_path}")
        if not set(np.unique(labels_np)).issubset({0.0, 1.0}):
            raise ValueError(f"Disease labels must be binary in {dynamic_path}")
        positive_indices = np.flatnonzero(labels_np == 1.0)
        if positive_indices.size and np.any(labels_np[positive_indices[0] :] != 1.0):
            raise ValueError(f"Disease labels must remain 1 after diagnosis in {dynamic_path}")

        labels = torch.from_numpy(labels_np)
        self.static_features.append(static)
        self.habits.append(torch.from_numpy(habits_np))
        self.labels.append(labels)
        self.valid_masks.append(diagnosis_prefix_mask(labels))
        self.ever_labels.append(float(labels.max().item() > 0.5))

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, index: int):
        return (
            self.pids[index],
            self.static_features[index],
            self.habits[index],
            self.labels[index],
            self.valid_masks[index],
            torch.tensor(self.ever_labels[index], dtype=torch.float32),
        )


def compute_pos_weights(
    dataset: LiverDataset, indices: Sequence[int]
) -> tuple[float, float]:
    selected = [int(index) for index in indices]
    event_positives = float(sum(dataset.ever_labels[index] for index in selected))
    event_negatives = float(len(selected) - event_positives)
    if event_positives == 0 or event_negatives == 0:
        raise ValueError("Training subset must contain both event classes")

    annual_positives = 0.0
    valid_count = 0.0
    for index in selected:
        valid = dataset.valid_masks[index]
        labels = dataset.labels[index]
        annual_positives += float(labels[valid].sum().item())
        valid_count += float(valid.sum().item())
    annual_negatives = valid_count - annual_positives
    if annual_positives == 0 or annual_negatives == 0:
        raise ValueError("Training subset must contain positive and negative at-risk years")
    return event_negatives / event_positives, annual_negatives / annual_positives


def build_stratified_folds(
    dataset: LiverDataset,
    n_splits: int = KFOLD,
    seed: int = DEFAULT_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    labels = np.asarray(dataset.ever_labels, dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2)
    if int(class_counts.min()) < n_splits:
        raise ValueError(
            f"Each event class needs at least {n_splits} patients; counts={class_counts.tolist()}"
        )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(dataset))
    return list(splitter.split(indices, labels))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = SEQ_LEN) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0).to(dtype=x.dtype)


def conditional_to_cumulative(hazards: torch.Tensor) -> torch.Tensor:
    if hazards.ndim < 1:
        raise ValueError("hazards must have at least one dimension")
    hazards = hazards.clamp(0.0, 1.0)
    return 1.0 - torch.cumprod(1.0 - hazards, dim=-1)


class TransLSTMMultiTask(nn.Module):
    def __init__(self, dropout: float = 0.1) -> None:
        super().__init__()
        self.static2h0 = nn.Linear(STATIC_DIM, EMBED_DIM * LSTM_LAYERS)
        self.habits_proj = nn.Linear(DYN_FEAT_DIM, EMBED_DIM)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=EMBED_DIM * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=TRANS_LAYERS,
            enable_nested_tensor=False,
        )
        self.pos_enc = PositionalEncoding(EMBED_DIM, max_len=SEQ_LEN)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=EMBED_DIM,
            hidden_size=EMBED_DIM,
            num_layers=LSTM_LAYERS,
            batch_first=True,
            dropout=dropout,
        )
        self.attn_vector = nn.Parameter(torch.empty(EMBED_DIM))
        nn.init.normal_(self.attn_vector, mean=0.0, std=EMBED_DIM**-0.5)
        self.time_head = nn.Linear(EMBED_DIM, 1)
        self.event_head = nn.Linear(EMBED_DIM, 1)

    @staticmethod
    def causal_mask(length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1
        )

    @staticmethod
    def _validate_mask(valid_mask: torch.Tensor, batch: int, length: int) -> None:
        if valid_mask.shape != (batch, length):
            raise ValueError(
                f"valid_mask must have shape {(batch, length)}, got {tuple(valid_mask.shape)}"
            )
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("every patient must have at least one valid year")
        if bool(((~valid_mask[:, :-1]) & valid_mask[:, 1:]).any()):
            raise ValueError("valid years must form a contiguous prefix")

    def forward(
        self,
        static_feat: torch.Tensor,
        habits: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ):
        if static_feat.ndim == 3 and static_feat.size(1) == 1:
            static_feat = static_feat.squeeze(1)
        if static_feat.ndim == 1:
            static_feat = static_feat.unsqueeze(0)
        if habits.ndim == 2:
            habits = habits.unsqueeze(0)
        if static_feat.ndim != 2 or static_feat.size(1) != STATIC_DIM:
            raise ValueError(f"static_feat must have shape [B, {STATIC_DIM}]")
        if habits.ndim != 3 or habits.size(2) != DYN_FEAT_DIM:
            raise ValueError(f"habits must have shape [B, T, {DYN_FEAT_DIM}]")

        batch, length, _ = habits.shape
        if static_feat.size(0) != batch:
            raise ValueError("static_feat and habits batch sizes differ")
        if valid_mask is None:
            valid_mask = torch.ones(batch, length, dtype=torch.bool, device=habits.device)
        else:
            valid_mask = valid_mask.to(device=habits.device, dtype=torch.bool)
        self._validate_mask(valid_mask, batch, length)

        h0 = self.static2h0(static_feat)
        h0 = h0.view(batch, LSTM_LAYERS, EMBED_DIM).transpose(0, 1).contiguous()
        c0 = torch.zeros_like(h0)

        x = self.dropout(self.pos_enc(self.habits_proj(habits)))
        x = self.transformer(
            x,
            mask=self.causal_mask(length, habits.device),
            src_key_padding_mask=~valid_mask,
        )

        lengths = valid_mask.sum(dim=1).to(dtype=torch.int64).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed, (h0, c0))
        out_seq, _ = pad_packed_sequence(
            packed_out, batch_first=True, total_length=length
        )

        time_logits = self.time_head(out_seq).squeeze(-1)
        attn_scores = torch.matmul(out_seq, self.attn_vector) / math.sqrt(EMBED_DIM)
        attn_scores = attn_scores.masked_fill(~valid_mask, torch.finfo(attn_scores.dtype).min)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_weights = attn_weights.masked_fill(~valid_mask, 0.0)
        event_rep = torch.sum(out_seq * attn_weights.unsqueeze(-1), dim=1)
        event_logits = self.event_head(event_rep).squeeze(-1)

        if return_attention:
            return time_logits, event_logits, attn_weights
        return time_logits, event_logits


def multitask_loss(
    annual_logits: torch.Tensor,
    event_logits: torch.Tensor,
    labels: torch.Tensor,
    valid_mask: torch.Tensor,
    ever_labels: torch.Tensor,
    event_pos_weight: float,
    annual_pos_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    valid_mask = valid_mask.to(dtype=torch.bool)
    annual_raw = nn.functional.binary_cross_entropy_with_logits(
        annual_logits,
        labels,
        reduction="none",
        pos_weight=torch.as_tensor(
            annual_pos_weight, dtype=annual_logits.dtype, device=annual_logits.device
        ),
    )
    annual_loss = annual_raw.masked_select(valid_mask).mean()
    event_loss = nn.functional.binary_cross_entropy_with_logits(
        event_logits,
        ever_labels,
        pos_weight=torch.as_tensor(
            event_pos_weight, dtype=event_logits.dtype, device=event_logits.device
        ),
    )
    total = annual_loss + event_loss
    return total, {"annual": annual_loss, "event": event_loss}


def make_optimizer(
    model: nn.Module,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
) -> tuple[Adam, ReduceLROnPlateau]:
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8,
        threshold=1e-4,
        min_lr=1e-7,
    )
    return optimizer, scheduler


def make_checkpoint(
    model: TransLSTMMultiTask,
    disease: str,
    training_metadata: dict,
) -> dict:
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    return {
        "format_version": 1,
        "architecture": "causal_transformer_lstm_attention_pooling",
        "disease": disease,
        "model_config": {
            "static_dim": STATIC_DIM,
            "dynamic_dim": DYN_FEAT_DIM,
            "embed_dim": EMBED_DIM,
            "num_heads": NUM_HEADS,
            "transformer_layers": TRANS_LAYERS,
            "lstm_layers": LSTM_LAYERS,
            "sequence_length": SEQ_LEN,
            "dropout": float(model.dropout.p),
            "causal_mask": True,
            "event_pooling": "learned_attention",
        },
        "lifestyle_columns": list(LIFESTYLE_COLUMNS),
        "risk_definition": "annual conditional hazard; cumulative=1-prod(1-hazard)",
        "model_state_dict": state,
        "training_metadata": training_metadata,
    }


def _move_batch(batch, device: torch.device):
    pids, static, habits, labels, valid, ever = batch
    return (
        list(pids),
        static.to(device=device, dtype=torch.float32, non_blocking=True),
        habits.to(device=device, dtype=torch.float32, non_blocking=True),
        labels.to(device=device, dtype=torch.float32, non_blocking=True),
        valid.to(device=device, dtype=torch.bool, non_blocking=True),
        ever.to(device=device, dtype=torch.float32, non_blocking=True),
    )


def train_epoch(
    model: TransLSTMMultiTask,
    loader: DataLoader,
    optimizer: Adam,
    device: torch.device,
    event_pos_weight: float,
    annual_pos_weight: float,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "annual_loss": 0.0, "event_loss": 0.0, "patients": 0}
    for batch in loader:
        _, static, habits, labels, valid, ever = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        annual_logits, event_logits = model(static, habits, valid)
        loss, parts = multitask_loss(
            annual_logits,
            event_logits,
            labels,
            valid,
            ever,
            event_pos_weight,
            annual_pos_weight,
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        batch_size = static.size(0)
        totals["loss"] += float(loss.detach()) * batch_size
        totals["annual_loss"] += float(parts["annual"].detach()) * batch_size
        totals["event_loss"] += float(parts["event"].detach()) * batch_size
        totals["patients"] += batch_size
    count = max(1, int(totals.pop("patients")))
    return {name: value / count for name, value in totals.items()}


@torch.no_grad()
def evaluate_epoch(
    model: TransLSTMMultiTask,
    loader: DataLoader,
    device: torch.device,
    event_pos_weight: float,
    annual_pos_weight: float,
) -> dict:
    model.eval()
    losses: list[tuple[float, float, float, int]] = []
    all_pids: list[str] = []
    all_hazards: list[np.ndarray] = []
    all_event_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_valid: list[np.ndarray] = []
    for batch in loader:
        pids, static, habits, labels, valid, ever = _move_batch(batch, device)
        annual_logits, event_logits = model(static, habits, valid)
        loss, parts = multitask_loss(
            annual_logits,
            event_logits,
            labels,
            valid,
            ever,
            event_pos_weight,
            annual_pos_weight,
        )
        size = static.size(0)
        losses.append((float(loss), float(parts["annual"]), float(parts["event"]), size))
        all_pids.extend(str(pid) for pid in pids)
        all_hazards.append(torch.sigmoid(annual_logits).cpu().numpy())
        all_event_probs.append(torch.sigmoid(event_logits).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_valid.append(valid.cpu().numpy())

    denominator = max(1, sum(item[3] for item in losses))
    hazards = np.concatenate(all_hazards, axis=0)
    return {
        "loss": sum(item[0] * item[3] for item in losses) / denominator,
        "annual_loss": sum(item[1] * item[3] for item in losses) / denominator,
        "event_loss": sum(item[2] * item[3] for item in losses) / denominator,
        "pids": all_pids,
        "hazards": hazards,
        "cumulative": 1.0 - np.cumprod(1.0 - hazards, axis=1),
        "event_probs": np.concatenate(all_event_probs, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "valid": np.concatenate(all_valid, axis=0),
    }


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return (
        float(roc_auc_score(labels, scores))
        if np.unique(labels).size == 2
        else float("nan")
    )


def calculate_prediction_metrics(result: dict) -> dict[str, float]:
    labels = result["labels"]
    hazards = result["hazards"]
    ever = (labels.max(axis=1) > 0.5).astype(np.int64)
    cumulative_15 = result["cumulative"][:, -1]
    metrics = {
        "cumulative_15y_auroc": _safe_auc(ever, cumulative_15),
        "cumulative_15y_auprc": (
            float(average_precision_score(ever, cumulative_15))
            if np.unique(ever).size == 2
            else float("nan")
        ),
        "event_head_auroc": _safe_auc(ever, result["event_probs"]),
    }

    first_year = np.where(ever == 1, np.argmax(labels, axis=1) + 1, 0)
    incident_counts = np.bincount(first_year[first_year > 0], minlength=SEQ_LEN + 1)[1:]
    annual_aucs = np.full(SEQ_LEN, np.nan, dtype=float)
    for year_index in range(SEQ_LEN):
        year = year_index + 1
        at_risk = (first_year == 0) | (first_year >= year)
        incident = (first_year == year).astype(np.int64)[at_risk]
        if np.unique(incident).size == 2:
            annual_aucs[year_index] = roc_auc_score(
                incident, hazards[at_risk, year_index]
            )
        metrics[f"year{year}_incident_auroc"] = float(annual_aucs[year_index])
    valid = (incident_counts > 0) & np.isfinite(annual_aucs)
    metrics["event_count_weighted_incident_auroc"] = (
        float(np.average(annual_aucs[valid], weights=incident_counts[valid]))
        if valid.any()
        else float("nan")
    )
    return metrics


def _loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def run_cross_validation(
    dataset: LiverDataset,
    disease: str,
    output_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    n_splits: int,
    seed: int,
    num_workers: int,
    max_folds: int | None = None,
) -> tuple[list[dict], list[dict], dict[str, float], dict]:
    folds = build_stratified_folds(dataset, n_splits=n_splits, seed=seed)
    epoch_rows: list[dict] = []
    fold_rows: list[dict] = []
    oof_rows: list[dict] = []
    global_best_auc = float("-inf")
    global_best_state = None
    global_best = None
    folds_to_run = folds if max_folds is None else folds[:max_folds]
    for fold_index, (train_indices, validation_indices) in enumerate(folds_to_run, start=1):
        fold_seed = seed + fold_index
        seed_everything(fold_seed)
        train_loader = _loader(
            Subset(dataset, train_indices), batch_size, True, fold_seed, num_workers
        )
        validation_loader = _loader(
            Subset(dataset, validation_indices), batch_size, False, fold_seed, num_workers
        )
        event_weight, annual_weight = compute_pos_weights(dataset, train_indices)
        model = TransLSTMMultiTask().to(device)
        optimizer, scheduler = make_optimizer(model)
        fold_best_auc = float("-inf")
        fold_best_epoch = 0
        fold_best_loss = float("nan")
        fold_best_state = None
        for epoch in range(1, epochs + 1):
            train_result = train_epoch(
                model,
                train_loader,
                optimizer,
                device,
                event_weight,
                annual_weight,
            )
            validation_result = evaluate_epoch(
                model,
                validation_loader,
                device,
                event_weight,
                annual_weight,
            )
            scheduler.step(validation_result["loss"])
            validation_metrics = calculate_prediction_metrics(validation_result)
            validation_auc = validation_metrics[
                "event_count_weighted_incident_auroc"
            ]
            epoch_rows.append(
                {
                    "disease": disease,
                    "fold": fold_index,
                    "epoch": epoch,
                    "train_loss": train_result["loss"],
                    "train_annual_loss": train_result["annual_loss"],
                    "train_event_loss": train_result["event_loss"],
                    "validation_loss": validation_result["loss"],
                    "validation_annual_loss": validation_result["annual_loss"],
                    "validation_event_loss": validation_result["event_loss"],
                    "Val_AUC": validation_auc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
            )
            if np.isfinite(validation_auc) and validation_auc > fold_best_auc:
                fold_best_auc = float(validation_auc)
                fold_best_epoch = epoch
                fold_best_loss = float(validation_result["loss"])
                fold_best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            if np.isfinite(validation_auc) and validation_auc > global_best_auc:
                global_best_auc = float(validation_auc)
                global_best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                global_best = {
                    "selected_fold": fold_index,
                    "selected_epoch": epoch,
                    "Val_AUC": float(validation_auc),
                    "validation_loss": float(validation_result["loss"]),
                    "training_patients": len(train_indices),
                    "validation_patients": len(validation_indices),
                    "training_ids": [dataset.pids[int(i)] for i in train_indices],
                    "validation_ids": [dataset.pids[int(i)] for i in validation_indices],
                }
            print(
                f"{disease} fold {fold_index}/{n_splits} epoch {epoch:03d}/{epochs} "
                f"train={train_result['loss']:.6f} val={validation_result['loss']:.6f} "
                f"Val_AUC={validation_auc:.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.2e}",
                flush=True,
            )

        if fold_best_state is None:
            raise RuntimeError(f"No checkpoint selected for {disease}, fold {fold_index}")
        model.load_state_dict(fold_best_state, strict=True)
        best_result = evaluate_epoch(
            model,
            validation_loader,
            device,
            event_weight,
            annual_weight,
        )
        fold_metrics = calculate_prediction_metrics(best_result)
        fold_rows.append(
            {
                "disease": disease,
                "fold": fold_index,
                "best_epoch": fold_best_epoch,
                "best_Val_AUC": fold_best_auc,
                "validation_loss_at_best_Val_AUC": fold_best_loss,
                "train_n": len(train_indices),
                "validation_n": len(validation_indices),
                **fold_metrics,
            }
        )
        for row_index, pid in enumerate(best_result["pids"]):
            row = {
                "ID": pid,
                "disease": disease,
                "fold": fold_index,
                "event_label": int(best_result["labels"][row_index].max() > 0.5),
                "event_head_probability": float(best_result["event_probs"][row_index]),
            }
            for year in range(1, SEQ_LEN + 1):
                row[f"Year{year}_conditional_risk"] = float(
                    best_result["hazards"][row_index, year - 1]
                )
                row[f"Year{year}_cumulative_risk"] = float(
                    best_result["cumulative"][row_index, year - 1]
                )
            oof_rows.append(row)

    pd.DataFrame(epoch_rows).to_csv(
        output_dir / "logs" / f"{disease}_cv_epoch_log.csv", index=False
    )
    pd.DataFrame(fold_rows).to_csv(
        output_dir / "logs" / f"{disease}_cv_fold_metrics.csv", index=False
    )
    pd.DataFrame(oof_rows).to_csv(
        output_dir / "oof_predictions" / f"{disease}_oof_predictions.csv", index=False
    )
    if max_folds is None:
        if len(oof_rows) != len(dataset):
            raise RuntimeError(
                f"OOF coverage error for {disease}: {len(oof_rows)} != {len(dataset)}"
            )
        ordered_oof = pd.DataFrame(oof_rows).set_index("ID").loc[dataset.pids]
        aggregate_result = {
            "hazards": ordered_oof[
                [f"Year{year}_conditional_risk" for year in range(1, SEQ_LEN + 1)]
            ].to_numpy(),
            "cumulative": ordered_oof[
                [f"Year{year}_cumulative_risk" for year in range(1, SEQ_LEN + 1)]
            ].to_numpy(),
            "event_probs": ordered_oof["event_head_probability"].to_numpy(),
            "labels": np.stack([label.numpy() for label in dataset.labels]),
        }
        aggregate_metrics = calculate_prediction_metrics(aggregate_result)
    else:
        aggregate_metrics = {}
    if global_best_state is None or global_best is None:
        raise RuntimeError(f"No finite Val_AUC checkpoint selected for {disease}")
    selected_model = TransLSTMMultiTask().to(device)
    selected_model.load_state_dict(global_best_state, strict=True)
    checkpoint = make_checkpoint(
        selected_model,
        disease,
        {
            "patients": len(dataset),
            "epochs": epochs,
            "cross_validation_folds": n_splits,
            "validation_fraction_per_fold": 1.0 / n_splits,
            "cross_validation_epochs_per_fold": epochs,
            "batch_size": batch_size,
            "learning_rate": LEARNING_RATE,
            "l2_weight_decay": WEIGHT_DECAY,
            "optimizer": "Adam",
            "scheduler": "ReduceLROnPlateau",
            "selection_metric": "event_count_weighted_incident_auroc",
            **global_best,
            "seed": seed,
            "static_directory": str(dataset.static_dir),
            "dynamic_directory": str(dataset.dynamic_dir),
            "oof_metrics": aggregate_metrics,
        },
    )
    model_path = output_dir / "models" / f"{disease}.pth"
    torch.save(checkpoint, model_path)
    return epoch_rows, fold_rows, aggregate_metrics, {
        **global_best,
        "model_path": str(model_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train 13 supplementary-method causal Transformer-LSTM models."
    )
    parser.add_argument(
        "--static-dir", default=r"G:\LiverDatasets\Static_Multimodal"
    )
    parser.add_argument(
        "--dynamic-dir", default=r"G:\LiverDatasets\NAFLD_corrected_v2"
    )
    parser.add_argument(
        "--output-dir",
        default=r"G:\LiverDatasets\Trans-LSTM-Supplementary-best-ValAUC-5fold-120ep-20260826",
    )
    parser.add_argument("--diseases", nargs="+", choices=DISEASES, default=DISEASES)
    parser.add_argument("--cv-epochs", type=int, default=EPOCHS)
    parser.add_argument("--folds", type=int, default=KFOLD)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cv_epochs < 1:
        raise ValueError("Epoch counts must be positive")
    output_dir = Path(args.output_dir)
    for subdirectory in ("models", "logs", "oof_predictions"):
        (output_dir / subdirectory).mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    all_metric_rows: list[dict] = []
    manifest = {
        "architecture": "128d, 2-layer causal Transformer, 2-layer LSTM, learned attention pooling",
        "risk_definition": "annual conditional hazard; cumulative=1-prod(1-hazard)",
        "static_features_unchanged": True,
        "arguments": vars(args),
        "models": [],
    }
    for disease_index, disease in enumerate(args.diseases):
        model_path = output_dir / "models" / f"{disease}.pth"
        if model_path.exists() and not args.overwrite:
            print(f"Skipping existing final model: {model_path}", flush=True)
            manifest["models"].append(str(model_path))
            continue
        dataset = LiverDataset(args.static_dir, args.dynamic_dir, disease)
        print(
            f"Starting {disease}: {len(dataset)} patients, "
            f"events={int(dataset.ever_labels.sum())}",
            flush=True,
        )
        _, _, cv_metrics, selection = run_cross_validation(
            dataset=dataset,
            disease=disease,
            output_dir=output_dir,
            device=device,
            epochs=args.cv_epochs,
            batch_size=args.batch_size,
            n_splits=args.folds,
            seed=args.seed + disease_index * 1000,
            num_workers=args.num_workers,
            max_folds=args.max_folds,
        )
        all_metric_rows.append({"disease": disease, **selection, **cv_metrics})
        manifest["models"].append(selection["model_path"])
        print(
            f"Saved best Val_AUC model: {selection['model_path']} "
            f"fold={selection['selected_fold']} epoch={selection['selected_epoch']} "
            f"Val_AUC={selection['Val_AUC']:.6f}",
            flush=True,
        )
    pd.DataFrame(all_metric_rows).to_csv(
        output_dir / "cross_validation_summary.csv", index=False
    )
    with (output_dir / "training_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
