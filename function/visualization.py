import os, glob, math
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # 화면 없는 서버용
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
    _USE_IMAGEIO = True
except Exception:
    _USE_IMAGEIO = False
    import cv2

from .data_loader import TEST_SEQUENCES, load_lidar_frame_polar, find_target_angle, filter_search_region, InferenceConfig

def _feats_from_rt(r, theta):
    """학습과 동일한 정규화로 (x,y,r,θ_norm) 생성 + x,y 반환"""
    x = r * np.cos(theta); y = r * np.sin(theta)
    feats = np.stack([x, y, r, theta], axis=1).astype(np.float32)
    r_max = max(r.max(), 1e-6)
    feats[:, 0:2] /= (r_max + 1e-6)
    feats[:, 2]   /= (r_max + 1e-6)
    feats[:, 3]   /= math.pi
    return feats, x, y

def _draw_frame(x, y, label, pred_mask_global,
                theta_center=None, search_angle=np.pi/3, search_radius=6.0,
                canvas_size=(900,900), lim=(8.0,8.0), title=""):
    """Matplotlib 한 프레임 그림 → RGB ndarray 반환"""
    w, h = canvas_size
    fig = plt.figure(figsize=(w/100, h/100), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    ax.set_xlim([-lim[0], lim[0]]); ax.set_ylim([-lim[1], lim[1]])
    ax.set_aspect('equal', adjustable='box'); ax.grid(True, alpha=0.2)

    # 모든 포인트(회색)
    ax.scatter(x, y, s=6, c="#BBBBBB", alpha=0.9, linewidths=0)

    # GT=1 (연두색 원형 테두리)
    tp = np.where(label == 1)[0]
    if len(tp) > 0:
        ax.scatter(x[tp], y[tp], s=18, facecolors="none", edgecolors="#00CC66", linewidths=1.2, label="GT=1")

    # Pred=1 (빨강)
    if pred_mask_global is not None and pred_mask_global.any():
        pi = np.where(pred_mask_global)[0]
        ax.scatter(x[pi], y[pi], s=12, c="#E74C3C", linewidths=0, label="Pred=1")

    # 검색 영역(부채꼴) + 중심선
    if theta_center is not None:
        th0 = theta_center - search_angle
        th1 = theta_center + search_angle
        t = np.linspace(th0, th1, 60)
        xr = np.concatenate([[0], search_radius*np.cos(t), [0]])
        yr = np.concatenate([[0], search_radius*np.sin(t), [0]])
        ax.fill(xr, yr, color="#3498DB", alpha=0.08)
        ax.plot([0, search_radius*np.cos(theta_center)],
                [0, search_radius*np.sin(theta_center)], color="#2980B9", lw=1.5, alpha=0.8)

    ax.set_title(title, fontsize=12)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return img


def _close_video_writer(writer):
    # imageio이면 close(), OpenCV이면 release()
    try:
        import imageio  # 존재 체크 용
        # imageio writer는 release가 없고 close만 있음
        if hasattr(writer, "close"):
            writer.close()
        else:
            writer.release()
    except Exception:
        # imageio가 없고 OpenCV 사용 중
        writer.release()


@torch.no_grad()
def render_sequence_to_video(seq_dir, model, device, out_path,
                             fps=10, prob_thresh=0.5,
                             use_search=True, search_angle_deg=60.0, search_radius=6.0,
                             xlim=8.0, ylim=8.0, canvas=(900,900)):
    """
    seq_dir: 시퀀스 폴더 (예: /root/seq-023-2dlidar_anot)
    model:   (B,N)->(B,N) 로짓을 내는 네트워크 (PointMLP/MAE 래퍼 등)
    """
    files = sorted(glob.glob(os.path.join(seq_dir, "*.txt")))
    if len(files) == 0:
        print(f"[WARN] no frames in {seq_dir}")
        return

    # 비디오 writer 준비
    if _USE_IMAGEIO:
        writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8)
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        w, h = canvas
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    prev_theta_center = None
    sr = math.radians(search_angle_deg)

    for t, f in enumerate(files):
        # 프레임 로드
        r, theta, label = load_lidar_frame_polar(f)
        feats, x, y = _feats_from_rt(r, theta)

        # 검색영역 서브셋 추출
        idx_subset = np.arange(len(r))
        if use_search and t > 0 and prev_theta_center is not None:
            _, _, _, idx_f = filter_search_region(r, theta, label,
                                                  prev_theta_center,
                                                  search_angle=sr,
                                                  search_radius=search_radius)
            if len(idx_f) > 0:
                idx_subset = idx_f

        # 모델 입력/예측
        inp = torch.from_numpy(feats[idx_subset]).unsqueeze(0).to(device)  # (1,M,4)
        logits = model(inp)                               # (1,M)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        pred_mask = (probs >= prob_thresh)

        # 전역 인덱스 기준 마스크
        pred_mask_global = np.zeros(len(r), dtype=bool)
        pred_mask_global[idx_subset[pred_mask]] = True

        # 다음 프레임 중심각 업데이트
        if t == 0:
            theta_gt_tp = theta[label == 1]
            base = theta_gt_tp if len(theta_gt_tp) > 0 else theta[idx_subset[pred_mask]]
            prev_theta_center = find_target_angle(base)
        else:
            prev_theta_center = find_target_angle(theta[idx_subset[pred_mask]])

        # 프레임 그리기
        title = f"{os.path.basename(seq_dir)} | {t+1}/{len(files)} | " \
                f"N={len(r)} M={len(idx_subset)} Det={int(pred_mask.sum())} | θc={prev_theta_center:.2f} rad"
        img = _draw_frame(
            x, y, label,
            pred_mask_global=pred_mask_global,
            theta_center=(prev_theta_center if use_search else None),
            search_angle=sr, search_radius=search_radius,
            canvas_size=canvas, lim=(xlim, ylim), title=title
        )

        # 비디오에 쓰기
        if _USE_IMAGEIO:
            writer.append_data(img)
        else:
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # writer.close()
    # print(f"[OK] saved: {out_path}")
    
    _close_video_writer(writer)
    print(f"[OK] saved: {out_path}")
    
    
def visualize_test_sequences(cfg, model, out_dir="videos",
                             fps=10, prob_thresh=0.5,
                             use_search=True, search_angle_deg=60.0, search_radius=6.0,
                             xlim=8.0, ylim=8.0, canvas=(900,900)):
    """
    TEST_SEQUENCES 리스트를 순회하며 시퀀스당 mp4 저장
    """
    os.makedirs(out_dir, exist_ok=True)
    for seq in TEST_SEQUENCES:
        seq_dir = os.path.join(cfg.root_dir, seq)
        if not os.path.isdir(seq_dir):
            print(f"[SKIP] {seq_dir} not found")
            continue
        out_path = os.path.join(out_dir, f"{seq}.mp4")
        render_sequence_to_video(
            seq_dir, model, cfg.device, out_path,
            fps=fps, prob_thresh=prob_thresh,
            use_search=use_search, search_angle_deg=search_angle_deg, search_radius=search_radius,
            xlim=xlim, ylim=ylim, canvas=canvas
        )
