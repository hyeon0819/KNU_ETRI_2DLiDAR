import os
import glob
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# 예외 시퀀스 (중간에 target person이 없어지는 경우가 있는 sequence)
EXCEPTION_SEQUENCES = ['seq-013-2dlidar_anot', 'seq-017-2dlidar_anot', 'seq-020-2dlidar_anot', 
                       'seq-026-2dlidar_anot', 'seq-027-2dlidar_anot']

TRAIN_SEQUENCES = ['seq-001-2dlidar_anot', 'seq-002-2dlidar_anot', 'seq-003-2dlidar_anot', 'seq-004-2dlidar_anot', 'seq-005-2dlidar_anot', 
                   'seq-006-2dlidar_anot', 'seq-007-2dlidar_anot', 'seq-008-2dlidar_anot', 'seq-009-2dlidar_anot', 'seq-010-2dlidar_anot',
                     'seq-011-2dlidar_anot', 'seq-012-2dlidar_anot', 'seq-013-2dlidar_anot', 'seq-014-2dlidar_anot', 'seq-015-2dlidar_anot',
                     'seq-016-2dlidar_anot', 'seq-017-2dlidar_anot', 'seq-018-2dlidar_anot', 'seq-019-2dlidar_anot', 'seq-020-2dlidar_anot',
                     'seq-021-2dlidar_anot', 'seq-022-2dlidar_anot']
TEST_SEQUENCES = ['seq-023-2dlidar_anot', 'seq-024-2dlidar_anot', 'seq-025-2dlidar_anot',
                  'seq-026-2dlidar_anot', 'seq-027-2dlidar_anot', 'seq-028-2dlidar_anot', 'seq-029-2dlidar_anot', 'seq-030-2dlidar_anot']

def load_lidar_frame_xy(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    각 프레임에서 (x, y, label) 데이터 load
    - 동일한 (x, y) 좌표가 있을 경우, label=1 (target person) 우선 처리
    """
    data = np.loadtxt(filepath, encoding='utf-8-sig')
    # Convert (x, y) to strings to merge numerically-identical locations robustly
    xy_str = np.char.add(data[:, 0].astype(str), '_')
    xy_str = np.char.add(xy_str, data[:, 1].astype(str))

    unique_xy, indices = np.unique(xy_str, return_inverse=True)

    # For each unique (x,y), choose the maximum label (prioritize 1)
    max_label_per_xy = np.zeros(len(unique_xy), dtype=int)
    for i, idx in enumerate(indices):
        max_label_per_xy[idx] = max(max_label_per_xy[idx], int(data[i, 2]))

    # Keep first occurrence per unique (x,y) whose label equals max label at that location
    selected = []
    seen = set()
    for i, idx in enumerate(indices):
        if idx not in seen:
            if int(data[i, 2]) == max_label_per_xy[idx]:
                selected.append(i)
                seen.add(idx)
    selected = np.array(selected, dtype=int)
    x = data[selected][:, 0]
    y = data[selected][:, 1]
    label = data[selected][:, 2].astype(int)
    return x, y, label


def load_lidar_frame_polar(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    (x, y) -> 극좌표로 읽기 (r, theta)
    """
    x, y, label = load_lidar_frame_xy(filepath)
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta, label


def find_target_angle(theta_tp: np.ndarray) -> float:
    if len(theta_tp) > 0:
        theta_tp_unwrapped = np.unwrap(theta_tp)
        theta_min_tp = np.min(theta_tp_unwrapped)
        theta_max_tp = np.max(theta_tp_unwrapped)
        theta_center = (theta_min_tp + theta_max_tp) / 2
    else:
        theta_center = 0.0
    return float(theta_center)


def filter_search_region(r: np.ndarray, theta: np.ndarray, label: np.ndarray,
                         theta_center: float, search_angle: float = math.pi/3, search_radius: float = 6.0):
    theta_min, theta_max = theta_center - search_angle, theta_center + search_angle
    idx = np.where((theta >= theta_min) & (theta <= theta_max) & (r <= search_radius))[0]
    return r[idx], theta[idx], label[idx], idx  # also return indices mapping into the original frame


@dataclass
class FrameRecord:
    """한 프레임의 메타정보(소속 시퀀스, 파일 경로, 프레임 인덱스)를 담는 작은 컨테이너."""
    seq: str
    path: str
    frame_idx: int


class LidarFramesDataset(Dataset):
    """
    Dataset Loader 클래스
    - feats: (N,4) = [x, y, r, theta] (옵션으로 프레임 단위 정규화)
    - labels: (N,)
    """
    def __init__(self, root_dir: str, sequences: List[str], use_polar: bool = True,
                 normalize: bool = True):
        self.root = root_dir
        self.use_polar = use_polar
        self.normalize = normalize
        self.frames: List[FrameRecord] = []

        for seq in sequences:
            seq_dir = os.path.join(root_dir, seq)
            files = sorted(glob.glob(os.path.join(seq_dir, '*.txt')))
            for i, f in enumerate(files):
                self.frames.append(FrameRecord(seq=seq, path=f, frame_idx=i))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.frames[idx]
        if self.use_polar:
            r, theta, label = load_lidar_frame_polar(rec.path)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
        else:
            x, y, label = load_lidar_frame_xy(rec.path)
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)

        feats = np.stack([x, y, r, theta], axis=1).astype(np.float32)
        labels = label.astype(np.float32)

        if self.normalize:
            # Simple per-frame normalization (helps stability for MLP)
            r_max = np.maximum(r.max(), 1e-6)
            feats[:, 0:2] /= (r_max + 1e-6)  # x, y
            feats[:, 2] /= (r_max + 1e-6)    # r
            feats[:, 3] /= math.pi           # θ in [-1,1]

        return {
            'feats': torch.from_numpy(feats),        # (N, 4)
            'labels': torch.from_numpy(labels),      # (N,)
            'seq': rec.seq,
            'frame_idx': torch.tensor(rec.frame_idx, dtype=torch.long)
        }


def pad_collate(batch: List[Dict[str, torch.Tensor]]):
    """
    DataLoader용 collate_fn:
    - 프레임마다 포인트 수(N)가 다르므로, 배치에서 가장 긴 N으로 padding
    - mask(True)가 유효 포인트를 표시 → 손실/평가 시 pad 무시 가능
    반환:
      feats : (B, max_N, 4)
      labels: (B, max_N)
      mask  : (B, max_N)  (True: 유효, False: pad)
      seqs  : 길이 B의 시퀀스명 리스트
      frame_idxs: (B,)
    """
    # Variable number of points per frame -> pad to max_N and provide mask
    max_N = max(item['feats'].shape[0] for item in batch)
    B = len(batch)

    feats = torch.zeros(B, max_N, 4, dtype=torch.float32)
    labels = torch.zeros(B, max_N, dtype=torch.float32)
    mask = torch.zeros(B, max_N, dtype=torch.bool)
    frame_idxs = torch.zeros(B, dtype=torch.long)

    seqs = []

    for b, item in enumerate(batch):
        N = item['feats'].shape[0]
        feats[b, :N] = item['feats']
        labels[b, :N] = item['labels']
        mask[b, :N] = True
        frame_idxs[b] = item['frame_idx']
        seqs.append(item['seq'])

    return {
        'feats': feats,      # (B, max_N, 4)
        'labels': labels,    # (B, max_N)
        'mask': mask,        # (B, max_N)
        'seqs': seqs,
        'frame_idxs': frame_idxs,
    }


# ----------------------------
# Metrics (PR / AP / F1)
# ----------------------------
@torch.no_grad()
def precision_recall_f1_from_logits(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
                                    thresh: float = 0.5) -> Dict[str, float]:
    probs = torch.sigmoid(logits)
    pred = (probs >= thresh).float()
    labels = labels.float()

    if mask is not None:
        pred = pred[mask]
        labels = labels[mask]

    tp = (pred * labels).sum().item()
    fp = (pred * (1 - labels)).sum().item()
    fn = ((1 - pred) * labels).sum().item()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {"precision": precision, "recall": recall, "f1": f1}


@torch.no_grad()
def average_precision_from_logits(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    # Compute PR curve by sweeping thresholds and do 11-point interpolation-like AP
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    if mask is not None:
        mask_np = mask.detach().cpu().numpy()
        probs = probs[mask_np]
        labels = labels[mask_np]

    # Sort by score desc
    order = np.argsort(-probs)
    probs = probs[order]
    labels = labels[order]

    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall = tp / np.maximum(labels.sum(), 1e-12)

    # AP via trapezoidal integration over recall
    # Ensure recall is non-decreasing
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[precision[0]], precision, [0.0]])
    # Make precision envelope
    for i in range(len(precision)-2, -1, -1):
        precision[i] = max(precision[i], precision[i+1])
    # Integrate
    ap = 0.0
    for i in range(1, len(recall)):
        ap += (recall[i] - recall[i-1]) * precision[i]
    return float(ap)


# ----------------------------
# Training / Evaluation
# ----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
    """
    학습/평가에 필요한 공통 설정 모음.
    - root_dir : 데이터 루트(시퀀스 폴더들을 포함)
    - weight_pos: 양성(레이블=1) 클래스 가중치(BCE pos_weight) → 클래스 불균형 완화
    """
    root_dir: str
    batch_size: int = 8
    lr: float = 1e-3
    epochs: int = 10
    weight_pos: float = 4.0  # class imbalance weight for positive (label=1)
    num_workers: int = 4
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def build_loaders(cfg: TrainConfig):
    train_ds = LidarFramesDataset(cfg.root_dir, TRAIN_SEQUENCES, use_polar=True)
    test_ds  = LidarFramesDataset(cfg.root_dir, TEST_SEQUENCES,  use_polar=True)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=pad_collate, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False,
                              num_workers=cfg.num_workers, collate_fn=pad_collate, pin_memory=True)
    return train_loader, test_loader


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    cfg: TrainConfig) -> Dict[str, float]:
    """Train for one epoch and aggregate metrics over variable-length frames safely.
    We collect *masked* logits/labels as 1D vectors so concatenation never depends
    on per-batch padded length.
    """
    model.train()
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.weight_pos], device=cfg.device))

    total_loss = 0.0
    # store masked, flattened vectors (1D)
    logits_vec, labels_vec = [], []

    for batch in loader:
        feats = batch['feats'].to(cfg.device)      # (B,N,4)
        labels = batch['labels'].to(cfg.device)    # (B,N)
        mask = batch['mask'].to(cfg.device)        # (B,N)

        logits = model(feats)                      # (B,N)
        loss = bce(logits[mask], labels[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # accumulate masked vectors
        logits_vec.append(logits[mask].detach())
        labels_vec.append(labels[mask].detach())

    logits_all = torch.cat(logits_vec, dim=0)  # (M,)
    labels_all = torch.cat(labels_vec, dim=0)  # (M,)

    # Build dummy mask (all True) for metric helpers expecting mask shape; set None to skip masking
    prf = precision_recall_f1_from_logits(logits_all, labels_all, mask=None, thresh=0.5)
    ap = average_precision_from_logits(logits_all, labels_all, mask=None)
    prf['ap'] = ap

    return {"loss": total_loss / max(len(loader), 1), **prf}


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, cfg: TrainConfig) -> Dict[str, float]:
    """Evaluate by aggregating masked vectors to avoid concat shape mismatches."""
    model.eval()
    logits_vec, labels_vec = [], []
    for batch in loader:
        feats = batch['feats'].to(cfg.device)
        labels = batch['labels'].to(cfg.device)
        mask = batch['mask'].to(cfg.device)
        logits = model(feats)
        logits_vec.append(logits[mask].detach())
        labels_vec.append(labels[mask].detach())

    if len(logits_vec) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "ap": 0.0}

    logits_all = torch.cat(logits_vec, dim=0)
    labels_all = torch.cat(labels_vec, dim=0)

    prf = precision_recall_f1_from_logits(logits_all, labels_all, mask=None, thresh=0.5)
    ap = average_precision_from_logits(logits_all, labels_all, mask=None)
    prf['ap'] = ap
    return prf

# ----------------------------
# Inference with Search Region (streaming per sequence)
# ----------------------------
@dataclass
class InferenceConfig:
    search_angle_deg: float = 60.0
    search_radius: float = 6.0
    prob_thresh: float = 0.5


@torch.no_grad()
def infer_sequence(model: nn.Module, seq_dir: str, device: str = 'cpu',
                  cfg: InferenceConfig = InferenceConfig()) -> List[Dict[str, np.ndarray]]:
    """
    Returns list per frame:
      {
        'idxs': indices in original frame (int array, length M),
        'probs': predicted probabilities for those indices (float array length M),
        'pred_mask': boolean mask indicating probs>=thresh among the M points,
        'theta_center': angle center used for this frame (float)
      }
    """
    model.eval()
    files = sorted(glob.glob(os.path.join(seq_dir, '*.txt')))
    prev_theta_center: Optional[float] = None
    last_pred_thetas = None
    results = []

    for t, f in enumerate(files):
        # Load full frame
        r, theta, label = load_lidar_frame_polar(f)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        feats_np = np.stack([x, y, r, theta], axis=1).astype(np.float32)

        # Normalize same as training
        r_max = max(r.max(), 1e-6)
        feats_np[:, 0:2] /= (r_max + 1e-6)
        feats_np[:, 2]   /= (r_max + 1e-6)
        feats_np[:, 3]   /= math.pi

        idx_subset = np.arange(len(r))

        # Apply search region from previous frame
        if t > 0 and prev_theta_center is not None:
            sr = math.radians(cfg.search_angle_deg)
            r_f, th_f, lab_f, idx_f = filter_search_region(r, theta, label,
                                                           prev_theta_center,
                                                           search_angle=sr,
                                                           search_radius=cfg.search_radius)
            if len(idx_f) > 0:
                feats_np = feats_np[idx_f]
                idx_subset = idx_f

        feats = torch.from_numpy(feats_np).to(device)
        logits = model(feats.unsqueeze(0))  # (1, M)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        pred_mask = probs >= cfg.prob_thresh
        theta_this = theta[idx_subset]
        theta_tp = theta_this[pred_mask]

        # Update search center for next frame
        if t == 0:
            # If first frame comes with labels available, use them; otherwise fallback to prediction
            theta_gt_tp = theta[label == 1]
            prev_theta_center = find_target_angle(theta_gt_tp if len(theta_gt_tp) > 0 else theta_tp)
        else:
            prev_theta_center = find_target_angle(theta_tp)

        results.append({
            'idxs': idx_subset,
            'probs': probs,
            'pred_mask': pred_mask,
            'theta_center': float(prev_theta_center) if prev_theta_center is not None else 0.0
        })

    return results
