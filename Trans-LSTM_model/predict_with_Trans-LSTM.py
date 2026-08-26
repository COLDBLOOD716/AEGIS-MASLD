from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


STATIC_DIM = 1024
DYN_FEAT_DIM = 10
EMBED_DIM = 128
NUM_HEADS = 4
TRANS_LAYERS = 2
LSTM_LAYERS = 2
SEQ_LEN = 15

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
        annual_logits = self.time_head(out_seq).squeeze(-1)
        attention_scores = torch.matmul(out_seq, self.attn_vector) / math.sqrt(EMBED_DIM)
        attention_scores = attention_scores.masked_fill(
            ~valid_mask, torch.finfo(attention_scores.dtype).min
        )
        attention = torch.softmax(attention_scores, dim=1)
        attention = attention.masked_fill(~valid_mask, 0.0)
        event_representation = torch.sum(out_seq * attention.unsqueeze(-1), dim=1)
        event_logits = self.event_head(event_representation).squeeze(-1)
        if return_attention:
            return annual_logits, event_logits, attention
        return annual_logits, event_logits


EXPECTED_CONFIG = {
    "static_dim": STATIC_DIM,
    "dynamic_dim": DYN_FEAT_DIM,
    "embed_dim": EMBED_DIM,
    "num_heads": NUM_HEADS,
    "transformer_layers": TRANS_LAYERS,
    "lstm_layers": LSTM_LAYERS,
    "sequence_length": SEQ_LEN,
    "causal_mask": True,
    "event_pooling": "learned_attention",
}


def load_model(
    model_path: str | os.PathLike[str], device: torch.device
) -> tuple[TransLSTMMultiTask, str, dict]:
    path = Path(model_path)
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(
            f"{path} is not a supplementary-method checkpoint with metadata"
        )
    config = checkpoint.get("model_config", {})
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in EXPECTED_CONFIG.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Checkpoint architecture mismatch in {path}: {mismatches}")
    if checkpoint.get("risk_definition") != "annual conditional hazard; cumulative=1-prod(1-hazard)":
        raise ValueError(f"Checkpoint risk definition is missing or incompatible: {path}")
    metadata = checkpoint.get("training_metadata", {})
    if metadata.get("selection_metric") != "event_count_weighted_incident_auroc":
        raise ValueError(f"Checkpoint was not selected by highest Val_AUC: {path}")
    required_selection_fields = (
        "selected_fold",
        "selected_epoch",
        "Val_AUC",
        "training_patients",
        "validation_patients",
    )
    missing_selection_fields = [
        field for field in required_selection_fields if field not in metadata
    ]
    if missing_selection_fields:
        raise ValueError(
            f"Checkpoint selection metadata is incomplete in {path}: {missing_selection_fields}"
        )
    dropout = float(config.get("dropout", 0.1))
    model = TransLSTMMultiTask(dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    disease = str(checkpoint.get("disease", path.stem))
    return model, disease, checkpoint


@torch.no_grad()
def predict_batch(
    model: TransLSTMMultiTask,
    static_features: torch.Tensor,
    habits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    static_features = static_features.to(device=device, dtype=torch.float32)
    habits = habits.to(device=device, dtype=torch.float32)
    annual_logits, event_logits = model(static_features, habits)
    hazards = torch.sigmoid(annual_logits)
    cumulative = conditional_to_cumulative(hazards)
    return hazards.cpu(), cumulative.cpu(), torch.sigmoid(event_logits).cpu()


class InferenceDataset(Dataset):
    def __init__(
        self,
        static_dir: str | os.PathLike[str],
        dynamic_dir: str | os.PathLike[str],
    ) -> None:
        self.static_dir = Path(static_dir)
        self.dynamic_dir = Path(dynamic_dir)
        if not self.static_dir.is_dir() or not self.dynamic_dir.is_dir():
            raise FileNotFoundError("static_dir and dynamic_dir must both exist")
        self.pids = sorted(
            path.name[: -len("-StaticMultimodal.pt")]
            for path in self.static_dir.glob("*-StaticMultimodal.pt")
        )
        if not self.pids:
            raise ValueError("No static multimodal features found")
        missing_dynamic = [
            pid for pid in self.pids if not (self.dynamic_dir / f"{pid}.csv").is_file()
        ]
        if missing_dynamic:
            raise ValueError(f"Missing dynamic CSVs for IDs: {missing_dynamic[:10]}")

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, index: int):
        pid = self.pids[index]
        static_path = self.static_dir / f"{pid}-StaticMultimodal.pt"
        raw_static = torch.load(static_path, map_location="cpu")
        if isinstance(raw_static, dict):
            if "feat" not in raw_static:
                raise ValueError(f"{static_path} is a dict without a 'feat' tensor")
            raw_static = raw_static["feat"]
        static = torch.as_tensor(raw_static, dtype=torch.float32).reshape(-1)
        if static.numel() != STATIC_DIM or not bool(torch.isfinite(static).all()):
            raise ValueError(f"Invalid 1024-dimensional static feature: {static_path}")

        dynamic_path = self.dynamic_dir / f"{pid}.csv"
        frame = pd.read_csv(dynamic_path, index_col=0)
        missing = [column for column in LIFESTYLE_COLUMNS if column not in frame.columns]
        if missing or len(frame) != SEQ_LEN:
            raise ValueError(
                f"Invalid dynamic CSV {dynamic_path}: missing={missing}, rows={len(frame)}"
            )
        array = frame[LIFESTYLE_COLUMNS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite lifestyle value in {dynamic_path}")
        if not set(np.unique(array)).issubset({0.0, 1.0}):
            raise ValueError(f"Lifestyle values must be binary in {dynamic_path}")
        return pid, static, torch.from_numpy(array)


def inference_all_diseases(
    static_dir: str | os.PathLike[str],
    dynamic_dir: str | os.PathLike[str],
    model_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    diseases: Sequence[str] = DISEASES,
    batch_size: int = 64,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
) -> tuple[Path, Path]:
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    patient_output_dir = output_dir / "per_patient"
    patient_output_dir.mkdir(parents=True, exist_ok=True)
    dataset = InferenceDataset(static_dir, dynamic_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    models: dict[str, TransLSTMMultiTask] = {}
    checkpoint_manifest: dict[str, str] = {}
    for requested_disease in diseases:
        path = model_dir / f"{requested_disease}.pth"
        if not path.is_file():
            raise FileNotFoundError(f"Missing required model: {path}")
        model, checkpoint_disease, _ = load_model(path, device)
        if checkpoint_disease != requested_disease:
            raise ValueError(
                f"Disease mismatch: filename={requested_disease}, checkpoint={checkpoint_disease}"
            )
        models[requested_disease] = model
        checkpoint_manifest[requested_disease] = str(path)

    summary_rows: list[dict] = []
    for pids, static, habits in loader:
        batch_predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for disease, model in models.items():
            hazards, cumulative, event_probs = predict_batch(model, static, habits)
            batch_predictions[disease] = (
                hazards.numpy(), cumulative.numpy(), event_probs.numpy()
            )
        for row_index, pid in enumerate(pids):
            frame_data: dict[str, list | np.ndarray] = {
                "Year": [f"Year{year}" for year in range(1, SEQ_LEN + 1)]
            }
            summary: dict[str, str | int | float] = {"ID": str(pid)}
            for disease in diseases:
                hazards, cumulative, event_probs = batch_predictions[disease]
                frame_data[f"{disease}_conditional_risk"] = hazards[row_index]
                frame_data[f"{disease}_cumulative_risk"] = cumulative[row_index]
                event_probability = float(event_probs[row_index])
                frame_data[f"{disease}_event_head_probability"] = [event_probability] * SEQ_LEN
                year15 = float(cumulative[row_index, -1])
                summary[f"{disease}_15y_cumulative_risk"] = year15
                summary[f"{disease}_15y_binary_at_0.5"] = int(year15 >= 0.5)
                summary[f"{disease}_event_head_probability"] = event_probability
            patient_frame = pd.DataFrame(frame_data)
            numeric_columns = patient_frame.columns.drop("Year")
            patient_frame[numeric_columns] = patient_frame[numeric_columns].round(8)
            patient_frame.to_csv(patient_output_dir / f"{pid}_pred.csv", index=False)
            summary_rows.append(summary)

    summary_path = output_dir / "prediction_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    manifest_path = output_dir / "inference_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "patients": len(dataset),
                "diseases": list(diseases),
                "models": checkpoint_manifest,
                "risk_definition": "annual conditional hazard; cumulative=1-prod(1-hazard)",
                "per_patient_directory": str(patient_output_dir),
                "summary": str(summary_path),
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    return summary_path, manifest_path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run supplementary-method causal Transformer-LSTM inference."
    )
    parser.add_argument("--static-dir", required=True)
    parser.add_argument("--dynamic-dir", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--diseases", nargs="+", choices=DISEASES, default=DISEASES)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary, manifest = inference_all_diseases(
        static_dir=args.static_dir,
        dynamic_dir=args.dynamic_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        diseases=args.diseases,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(f"Saved prediction summary: {summary}")
    print(f"Saved inference manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
