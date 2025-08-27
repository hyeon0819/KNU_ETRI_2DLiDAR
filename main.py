import os
import glob
import math
import argparse
import logging

import numpy as np
import torch

from function.utils import save_test_sequences_detections
from function.data_loader import (
    TrainConfig, set_seed, build_loaders,
    train_one_epoch, evaluate,
    load_lidar_frame_xy,
    TRAIN_SEQUENCES, TEST_SEQUENCES,
)
from models.point_mae import PointMAE2DLiDARDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Train 2D LiDAR detector; save detections once at the end using best model.")
    # 필수
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")

    # 선택
    parser.add_argument("--data_root", type=str, default="./data/annotations",
                        help="Root dir with seq-XXX-2dlidar_anot folders")
    parser.add_argument("--num_epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_pos", type=float, default=4.0)
    parser.add_argument("--device", type=int, default=0, choices=[0, 1])
    parser.add_argument("--num_group", type=int, default=256)
    parser.add_argument("--group_size", type=int, default=64)

    # 마지막에 딱 한 번 저장할 detection 설정
    parser.add_argument("--save_detections", action="store_true",
                        help="If set, after training finishes, load best model and export detections once.")

    return parser.parse_args()



# -----------------------------
# Logging
# -----------------------------
def setup_logger(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(output_dir, "training.log"),
        filemode="a",
        level=logging.INFO,
        format='[%(asctime)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

def print_log(msg: str, logger: logging.Logger = None):
    print(msg)
    if logger is not None:
        logger.info(msg)
        
        
# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    print_log('---------------------------------------')
    print_log(f"Arguments: {args}")
    print_log('---------------------------------------')

    device = torch.device(f"cuda:{args.device}") if torch.cuda.is_available() else torch.device("cpu")
    # logger
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(output_dir)

    # seed & data
    set_seed(42)
    cfg = TrainConfig(
        root_dir=args.data_root,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.num_epoch,
        weight_pos=args.weight_pos,
        device=device,
    )
    train_loader, test_loader = build_loaders(cfg)

    # model & optim
    model = PointMAE2DLiDARDetector(
        group_size=args.group_size, num_group=args.num_group,
        encoder_dims=256, trans_dim=256,
        depth=4, num_heads=8, drop_path_rate=0.1,
        point_head_hidden=128, scatter_reduce='mean'
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    # ---- Training loop ----
    best_f1 = -1.0  # 첫 epoch에 무조건 저장되도록
    ckpt_path = os.path.join(output_dir, "best_model.pth")

    print_log(f"Training started.")
    for e in range(cfg.epochs):
        tr = train_one_epoch(model, train_loader, optimizer, cfg)
        ev = evaluate(model, test_loader, cfg)

        print_log(
            (f"[Epoch {e+1:03d}] "
             f"loss={tr['loss']:.4f} | TR p={tr['precision']:.3f} r={tr['recall']:.3f} f1={tr['f1']:.3f} ap={tr['ap']:.3f} | "
             f"EV p={ev['precision']:.3f} r={ev['recall']:.3f} f1={ev['f1']:.3f} ap={ev['ap']:.3f}"),
            logger
        )

        # 원래 코드처럼 test 한 번 더
        test_ev = evaluate(model, test_loader, cfg)
        print_log(f"-----Testing----- TestEV p={test_ev['precision']:.3f} r={test_ev['recall']:.3f} f1={test_ev['f1']:.3f} ap={test_ev['ap']:.3f}", logger)

        # 베스트 갱신이면 저장 (마지막에 이 체크포인트로 탐지 1회 수행)
        if test_ev['f1'] > best_f1:
            best_f1 = test_ev['f1']
            torch.save(model.state_dict(), ckpt_path)
            print_log(f"Saved best model with F1: {best_f1:.3f} → {ckpt_path}", logger)

    print_log(f"Training finished. Best F1: {best_f1:.3f}", logger)

    # ---- After training: load best & save detections once ----
    if args.save_detections:
        if os.path.isfile(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print_log(f"[POST] Loaded best model from {ckpt_path}", logger)

            det_root = os.path.join(output_dir, "detections_best")
            save_test_sequences_detections(
                cfg, model, out_dir=det_root,
                prob_thresh=0.5,
                use_search=False,          # 시각화처럼 search region 쓰고 싶으면 True
                search_angle_deg=60.0,
                search_radius=6.0
            )
        else:
            print_log(f"[WARN] Best checkpoint not found: {ckpt_path}", logger)


if __name__ == "__main__":
    main()