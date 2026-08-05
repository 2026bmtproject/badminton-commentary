"""
MMPose 骨架抽取 -> 骨架 CSV
============================
讀取「影片 + 球場邊界」，用 MMPose 對每幀做 2D Human Pose Estimation，
再用球場 homography 選出場內 Top / Bottom 兩名球員，輸出逐幀骨架 CSV。

輸出 CSV（可直接餵給 bst_infer_standalone.py）：
    frame, player(top/bottom), det_idx,
    bbox_x1, bbox_y1, bbox_x2, bbox_y2,
    <joint>_x, <joint>_y, <joint>_s   （17 個 COCO 關節，原始像素座標）
每幀固定兩列（top / bottom）；某幀選不到該位球員時該列座標留空。


依賴（pip 套件）：mmpose, mmdet, mmcv, opencv-python, numpy, pandas, tqdm

用法：
    python mmpose_skeleton_to_csv.py --video clip.mp4 --court court.json --out clip_skeleton.csv

    # 批量：對資料夾內每支影片各輸出 <影片名>_skeleton.csv（共用同一 court）
    python mmpose_skeleton_to_csv.py --video-dir clips/ --court court.csv --out-dir out/
"""

import csv
import json
import argparse
import pickle
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm


# ====================================================================
# court_to_homography（內嵌）
# ====================================================================
COURT_BOUNDS_NORMALIZED = {
    'border_L': 0.0, 'border_R': 1.0, 'border_U': 0.0, 'border_D': 1.0,
}


def order_corners(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    s = p[np.argsort(ang)]
    tl = int(np.argmin(s.sum(axis=1)))
    s = np.roll(s, -tl, axis=0)
    if s[1, 1] < s[3, 1]:
        s = s[[0, 3, 2, 1]]
    return s


def extract_outer_corners(points: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float32)
    hull = cv2.convexHull(pts)
    peri = cv2.arcLength(hull, True)
    approx = None
    for k in (0.02, 0.03, 0.05, 0.08, 0.1):
        cand = cv2.approxPolyDP(hull, k * peri, True).reshape(-1, 2)
        if len(cand) == 4:
            approx = cand
            break
    if approx is None or len(approx) != 4:
        raise ValueError(
            f"無法從 court 點雲抓出 4 個外框角點（得到 "
            f"{0 if approx is None else len(approx)} 點）。請確認 court CSV 是完整球場邊界。"
        )
    return order_corners(approx.astype(np.float64))


def compute_homography(corners: np.ndarray) -> np.ndarray:
    court = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float64)
    H, _ = cv2.findHomography(corners, court)
    if H is None:
        raise ValueError("cv2.findHomography 失敗，角點可能共線或順序錯誤。")
    return H


def load_court_points(court_path: str) -> np.ndarray:
    """讀取球場點；支援舊式分號 CSV 與 court detector JSON。"""
    path = Path(court_path)
    if path.suffix.lower() == '.json':
        with path.open(encoding='utf-8') as f:
            data = json.load(f)

        courts = data.get('courts')
        if not isinstance(courts, list) or not courts:
            raise ValueError("court JSON 必須包含非空的 'courts' 陣列。")

        corners = courts[0].get('corners') if isinstance(courts[0], dict) else None
        points = np.asarray(corners, dtype=np.float64)
        if points.shape != (4, 2):
            raise ValueError(
                "court JSON 的 courts[0].corners 應為 (4, 2)，"
                f"實際 {points.shape}"
            )
        return points

    points = np.loadtxt(path, delimiter=';')
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"court CSV 應為 (N, 2)，實際 {points.shape}")
    return points


def load_court_homography(court_path: str):
    points = load_court_points(court_path)
    corners = extract_outer_corners(points)
    H = compute_homography(corners)
    return H, dict(COURT_BOUNDS_NORMALIZED)


# ====================================================================
# COCO-17 關節名稱
# ====================================================================
COCO_NAMES = [
    'nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
    'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow',
    'L_wrist', 'R_wrist', 'L_hip', 'R_hip',
    'L_knee', 'R_knee', 'L_ankle', 'R_ankle',
]


# ====================================================================
# Step 1: 跑 MMPose 抽骨架（含信心分數），結果可快取成 .pkl
# ====================================================================
def extract_skeletons(video_path, cache_path=None, get_inferencer=None,
                      device=None):
    """回傳每幀 dict: {'kps': (n,17,2), 'scores': (n,17), 'bboxes': (n,4)}。"""
    if cache_path and Path(cache_path).exists():
        print(f"[cache] 讀取既有骨架快取: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    if get_inferencer is not None:
        inferencer = get_inferencer()
    else:
        from mmpose.apis import MMPoseInferencer
        print("[mmpose] 載入 inferencer 'human' ...")
        inferencer = MMPoseInferencer('human', device=device)

    frames = []
    print("[mmpose] 逐幀抽取骨架（第一次會比較久）...")
    for result in tqdm(inferencer(video_path, show=False)):
        preds = result['predictions'][0]
        if len(preds) == 0:
            frames.append({
                'kps': np.zeros((0, 17, 2), np.float32),
                'scores': np.zeros((0, 17), np.float32),
                'bboxes': np.zeros((0, 4), np.float32),
            })
            continue
        kps = np.array([p['keypoints'] for p in preds], np.float32)        # (n,17,2)
        scr = np.array([p['keypoint_scores'] for p in preds], np.float32)  # (n,17)
        bbs = np.array([p['bbox'][0] for p in preds], np.float32)          # (n,4)
        frames.append({'kps': kps, 'scores': scr, 'bboxes': bbs})

    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump(frames, f)
        print(f"[cache] 骨架已快取到: {cache_path}")
    return frames


# ====================================================================
# Step 2: 用球場 homography 選出場內 Top / Bottom 兩名球員
# ====================================================================
def select_in_court(frame_det, H, court_bounds, eps=0.01, y_margin=0.12):
    """回傳 (top_idx, bottom_idx)；選不到回 (None, None)。idx 是 frame_det['kps'] 內的列索引。

    y_margin: 上下（底線方向）額外放寬的比例，讓起跳殺球（雙腳離地、homography 落點被推到
        底線外）的球員不被誤排除；左右（邊線方向）仍只用很小的 eps，避免抓到邊線外的裁判。
    """
    kps = frame_det['kps']
    if len(kps) < 2:
        return None, None

    feet = kps[:, -2:, :]           # (n, 2, 2) 兩腳踝
    feet_mid = feet.mean(axis=1)    # (n, 2)
    feet_cam = feet_mid.T           # (2, n)
    feet_cam_h = np.vstack([feet_cam, np.ones((1, feet_cam.shape[1]))])
    feet_court = H @ feet_cam_h
    feet_court = feet_court[:2] / feet_court[2:]  # (2, n)

    xL, xR = court_bounds['border_L'], court_bounds['border_R']
    yU, yD = court_bounds['border_U'], court_bounds['border_D']
    xn = (feet_court[0] - xL) / (xR - xL)
    yn = (feet_court[1] - yU) / (yD - yU)

    in_court = (xn > -eps) & (xn < 1 + eps) & (yn > -y_margin) & (yn < 1 + y_margin)
    ids = np.nonzero(in_court)[0]
    if len(ids) < 2:
        return None, None

    pid = ids[:2]
    if yn[pid[0]] > yn[pid[1]]:  # y 較小的是 Top
        pid = pid[::-1]
    return int(pid[0]), int(pid[1])


# ====================================================================
# Step 3: 輸出骨架 CSV
# ====================================================================
def write_skeleton_csv(out_path, dets, sel):
    header = ['frame', 'player', 'det_idx', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
    for name in COCO_NAMES:
        header += [f'{name}_x', f'{name}_y', f'{name}_s']

    def player_row(frame_idx, label, idx, det):
        row = [frame_idx, label, '' if idx is None else idx]
        if idx is None:
            row += [''] * 4              # bbox
            row += [''] * (17 * 3)       # joints
            return row
        bb = det['bboxes'][idx]
        row += [f'{v:.2f}' for v in bb[:4]]
        kps = det['kps'][idx]
        scr = det['scores'][idx]
        for j in range(17):
            row += [f'{kps[j, 0]:.2f}', f'{kps[j, 1]:.2f}', f'{scr[j]:.4f}']
        return row

    n = len(dets)
    n_top = n_bottom = 0
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i in range(n):
            det = dets[i]
            top_idx, bottom_idx = sel[i]
            n_top += top_idx is not None
            n_bottom += bottom_idx is not None
            writer.writerow(player_row(i, 'top', top_idx, det))
            writer.writerow(player_row(i, 'bottom', bottom_idx, det))
    print(f"[out] 完成: {out_path}  (共 {n} 幀 × 2 列; "
          f"top 有效 {n_top}/{n} 幀, bottom 有效 {n_bottom}/{n} 幀)")


# ====================================================================
# 單支影片處理
# ====================================================================
def process_video(video_path, court_data, out_path, y_margin=0.12,
                  cache_path=None, get_inferencer=None, device=None):
    H, bounds = court_data
    dets = extract_skeletons(
        video_path, cache_path, get_inferencer=get_inferencer,
        device=device)
    sel = [select_in_court(d, H, bounds, y_margin=y_margin) for d in dets]
    write_skeleton_csv(out_path, dets, sel)


VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')


def main():
    ap = argparse.ArgumentParser(
        description='MMPose 抽骨架 + 球場選人 -> 骨架 CSV（無標記影片輸出）')
    ap.add_argument('--video', help='輸入影片（單支模式）')
    ap.add_argument('--video-dir', dest='video_dir', help='輸入影片資料夾（批量模式）')
    ap.add_argument(
        '--court', required=True,
        help='球場邊界 JSON（court.json）或分號 CSV（selected_court.csv）')
    ap.add_argument('--out', help='輸出骨架 CSV 路徑（單支模式；預設 <影片名>_skeleton.csv）')
    ap.add_argument('--out-dir', dest='out_dir',
                    help='批量模式輸出資料夾（預設與 --video-dir 相同）')
    ap.add_argument('--y-margin', dest='y_margin', type=float, default=0.12,
                    help='選人時上下（底線方向）額外放寬比例，讓起跳球員不被排除（預設 0.12）')
    ap.add_argument('--cache', help='骨架快取 .pkl 路徑（單支模式；預設 <影片名>_pose.pkl）')
    ap.add_argument('--no-cache', action='store_true', help='不讀寫骨架快取 .pkl')
    ap.add_argument(
        '--device', choices=('cpu', 'cuda'), default=None,
        help='MMPose 執行裝置（預設由框架自動選擇）')
    args = ap.parse_args()

    if bool(args.video) == bool(args.video_dir):
        ap.error('請擇一提供 --video（單支）或 --video-dir（批量）。')

    # 球場 homography 只讀一次（批量共用）
    court_data = load_court_homography(args.court)
    print(f"[court] 已載入球場 homography: {args.court}")

    # ---------------- 批量模式 ----------------
    if args.video_dir:
        video_dir = Path(args.video_dir)
        if not video_dir.is_dir():
            ap.error(f'--video-dir 不是資料夾: {video_dir}')
        videos = sorted(p for p in video_dir.iterdir()
                        if p.is_file() and p.suffix in VIDEO_EXTS)
        if not videos:
            ap.error(f'{video_dir} 內找不到任何影片')
        out_dir = Path(args.out_dir) if args.out_dir else video_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        # MMPose inferencer 整批只載入一次共用，避免累積 GPU 記憶體
        _inf = {'obj': None}

        def get_inferencer():
            if _inf['obj'] is None:
                from mmpose.apis import MMPoseInferencer
                print("[mmpose] 載入 inferencer 'human'（整批共用，只載入一次）...")
                _inf['obj'] = MMPoseInferencer('human', device=args.device)
            return _inf['obj']

        print(f"[batch] 共 {len(videos)} 支影片，輸出到 {out_dir}")
        for n, video in enumerate(videos, 1):
            print(f"\n[batch] ({n}/{len(videos)}) {video.name}")
            cache = None if args.no_cache else str(video.with_suffix('')) + '_pose.pkl'
            out_path = str(out_dir / f'{video.stem}_skeleton.csv')
            process_video(str(video), court_data, out_path,
                          y_margin=args.y_margin, cache_path=cache,
                          get_inferencer=get_inferencer, device=args.device)
        print(f"\n[batch] 全部完成，共處理 {len(videos)} 支影片。")
        return

    # ---------------- 單支模式 ----------------
    out_path = args.out or (str(Path(args.video).with_suffix('')) + '_skeleton.csv')
    cache = None if args.no_cache else (args.cache or str(Path(args.video).with_suffix('')) + '_pose.pkl')
    process_video(args.video, court_data, out_path,
                  y_margin=args.y_margin, cache_path=cache,
                  device=args.device)


if __name__ == '__main__':
    main()
