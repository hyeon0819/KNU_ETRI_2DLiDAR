import os, glob, math
import numpy as np
import torch
import csv

from .data_loader import TEST_SEQUENCES, load_lidar_frame_polar, find_target_angle, \
                    filter_search_region, InferenceConfig


def _feats_from_rt(r, theta):
    """학습과 동일한 정규화로 (x,y,r,θ_norm) 생성 + x,y 반환"""
    x = r * np.cos(theta); y = r * np.sin(theta)
    feats = np.stack([x, y, r, theta], axis=1).astype(np.float32)
    r_max = max(r.max(), 1e-6)
    feats[:, 0:2] /= (r_max + 1e-6)
    feats[:, 2]   /= (r_max + 1e-6)
    feats[:, 3]   /= math.pi
    return feats, x, y


@torch.no_grad()
def save_sequence_detections(
    seq_dir, model, device, out_csv,
    prob_thresh=0.5,
    use_search=False, search_angle_deg=60.0, search_radius=6.0,
):
    """
    시퀀스 하나(seq_dir)에 대해 detection 결과를 CSV로 저장.
    CSV 컬럼: frame_idx, point_idx, x, y, prob, pred, gt

    - 시각화 코드와 동일하게: (r,theta) -> feats=(x,y,r,theta_norm) 생성
    - use_search=True면, 이전 프레임 탐지 중심각을 기준으로 부채꼴 영역에서만 재탐색
    """
    files = sorted(glob.glob(os.path.join(seq_dir, "*.txt")))
    if len(files) == 0:
        print(f"[WARN] no frames in {seq_dir}")
        return

    # 검색영역 각도(rad)
    sr = math.radians(search_angle_deg)
    prev_theta_center = None

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_idx", "point_idx", "x", "y", "prob", "pred", "gt"])

        for t, fpath in enumerate(files):
            # 1) 프레임 로드 (r,theta,label) -> (feats,x,y)
            r, theta, label = load_lidar_frame_polar(fpath)
            feats, x, y = _feats_from_rt(r, theta)

            # 2) 검색 영역 적용(옵션)
            idx_subset = np.arange(len(r))
            if use_search and t > 0 and prev_theta_center is not None:
                _, _, _, idx_f = filter_search_region(
                    r, theta, label,
                    prev_theta_center,
                    search_angle=sr,
                    search_radius=search_radius
                )
                if len(idx_f) > 0:
                    idx_subset = idx_f

            # 3) 모델 추론
            if len(idx_subset) > 0:
                inp = torch.from_numpy(feats[idx_subset]).unsqueeze(0).to(device)  # (1,M,4)
                logits = model(inp)                                               # (1,M)
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                pred_mask = (probs >= prob_thresh)
            else:
                # 서브셋이 비면 아무것도 쓰지 않고 다음 프레임으로
                probs = np.zeros(0, dtype=np.float32)
                pred_mask = np.zeros(0, dtype=bool)

            # 4) 다음 프레임 탐색 중심각 갱신
            if use_search:
                if t == 0:
                    theta_gt_tp = theta[label == 1]
                    base = theta_gt_tp if len(theta_gt_tp) > 0 else theta[idx_subset[pred_mask]] if len(idx_subset) > 0 else np.array([])
                    prev_theta_center = find_target_angle(base)
                else:
                    base = theta[idx_subset[pred_mask]] if len(idx_subset) > 0 else np.array([])
                    prev_theta_center = find_target_angle(base)

            # 5) CSV 저장 (전역 인덱스 기준으로 기록)
            if len(idx_subset) > 0:
                for local_i, gi in enumerate(idx_subset):
                    p = float(probs[local_i])
                    pred = int(pred_mask[local_i])
                    gt = int(label[gi])
                    writer.writerow([t, gi, f"{x[gi]:.6f}", f"{y[gi]:.6f}", f"{p:.6f}", pred, gt])

    print(f"[OK] detections saved: {out_csv}")


@torch.no_grad()
def save_test_sequences_detections(
    cfg, model, out_dir="detections",
    prob_thresh=0.5,
    use_search=False, search_angle_deg=60.0, search_radius=6.0
):
    """
    TEST_SEQUENCES 전체에 대해, 시퀀스별 CSV 저장.
    파일 경로: {out_dir}/{seq}.csv
    """
    os.makedirs(out_dir, exist_ok=True)

    for seq in TEST_SEQUENCES:
        seq_dir = os.path.join(cfg.root_dir, seq)
        if not os.path.isdir(seq_dir):
            print(f"[SKIP] {seq_dir} not found")
            continue
        out_csv = os.path.join(out_dir, f"{seq}.csv")
        save_sequence_detections(
            seq_dir, model, cfg.device, out_csv,
            prob_thresh=prob_thresh,
            use_search=use_search, search_angle_deg=search_angle_deg, search_radius=search_radius
        )