"""
Skeleton Visualization / Sanity-Check
=====================================
把 MMPose 偵測到的 COCO-17 骨架直接畫回原始影片,讓你用眼睛確認骨架標記是否準確。

它畫的是「未經正規化的原始像素座標」,也就是 MMPose 真正偵測到的位置;同時(若提供
球場 homography)用 bst_infer_from_video.py 完全相同的選人邏輯標出哪兩個人被選為
場內球員(Top 在前),讓你確認「模型真正吃到的那兩具骨架」對不對。

畫面說明:
  - 亮綠色骨架 = 被選為 Top player(球場上方,y 較小)
  - 亮藍色骨架 = 被選為 Bottom player(球場下方)
  - 灰色細線   = 偵測到但沒被選中的人(場外 / 多餘偵測)
  - 紅色小點   = 信心低於門檻的關節(--kp-thr)
  - 黃色大圈   = 兩手腕(擊球關鍵點),方便檢查
  - 紅色圓點   = 球(TrackNet CSV)
  - 上方橫幅   = 該幀是擊球影格(來自 hit_frames)

依賴:
    pip install mmpose mmdet opencv-python numpy pandas tqdm
    (court / shuttle / hit-frames 都是選用,不給就只畫所有偵測到的骨架)

用法:
    # 1) 輸出一支標註好的影片(最常用)
    python viz_skeleton.py --video test.mp4 --save test_skeleton.mp4 --court court.json --shuttle test_ball.csv --hits hit_frames.csv

    # 2) 互動逐幀檢視(需要 GUI 視窗)
    python viz_skeleton.py --video b_seg0028.mp4 --show --court court.json

    # 3) 只想快速看骨架,不管球場/球
    python viz_skeleton.py --video test.mp4 --save out.mp4

    # 4) 輸出場內 Top/Bottom 兩名球員骨架 CSV(原始像素座標,需要 --court)
    python viz_skeleton.py --video b_seg0016.mp4 --court court.json --save-csv 28_skeleton.csv

    # 5) 批量處理整個資料夾(用法一:標註影片)。全部影片共用同一個 --court（支援 court.json 或舊版 CSV）。
    #    輸出檔名為 <影片名>_skeleton.mp4,放到 --out-dir(預設與 --video-dir 同)。
    python viz_skeleton.py --video-dir MK_vs_CT_2019/ --out-dir MK_vs_CT_2019_skeleton/ --court court.json --save

    # 6) 批量處理整個資料夾(用法四:骨架 CSV)。全部影片共用同一個 --court（支援 court.json 或舊版 CSV）。
    #    輸出檔名為 <影片名>_skeleton.csv。
    python viz_skeleton.py --video-dir ASY_vs_PC_2021_fregment/ --court ASY_vs_PC_2021_court.csv --save-csv

    #    批量模式下 shuttle/hits 為選用,給資料夾會依影片檔名自動配對:
    #      --shuttle-dir balls/ --hits-dir hits/

互動模式按鍵:
    空白    播放 / 暫停
    → / d   下一幀        ← / a   上一幀
    . / e   下 10 幀      , / q   上 10 幀
    n       下一個擊球幀  p       上一個擊球幀
    s       存下目前這一幀為 png
    Esc     離開
"""

import sys
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

# 讓 import 找得到 court_to_homography(與本檔同目錄)
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ====================================================================
# COCO-17 關節 / 骨架連線定義
# ====================================================================
COCO_NAMES = [
    'nose', 'L_eye', 'R_eye', 'L_ear', 'R_ear',
    'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow',
    'L_wrist', 'R_wrist', 'L_hip', 'R_hip',
    'L_knee', 'R_knee', 'L_ankle', 'R_ankle',
]
WRIST_IDS = (9, 10)
ANKLE_IDS = (15, 16)

# (a, b, 屬於左/右/中) — 用來決定線的顏色(左:暖色,右:冷色,中:白)
BONES = [
    (0, 1, 'c'), (0, 2, 'c'), (1, 3, 'l'), (2, 4, 'r'),
    (3, 5, 'l'), (4, 6, 'r'),
    (5, 7, 'l'), (7, 9, 'l'), (6, 8, 'r'), (8, 10, 'r'),
    (5, 6, 'c'), (5, 11, 'l'), (6, 12, 'r'), (11, 12, 'c'),
    (11, 13, 'l'), (13, 15, 'l'), (12, 14, 'r'), (14, 16, 'r'),
]


# ====================================================================
# Step 1: 跑 MMPose 抽骨架(含信心分數),結果快取成 .pkl
# ====================================================================
def extract_skeletons(video_path: str, cache_path: str | None = None,
                      get_inferencer=None):
    """
    回傳每幀的偵測結果 list,每個元素是 dict:
        {'kps': (n, 17, 2), 'scores': (n, 17), 'bboxes': (n, 4)}
    若 cache_path 存在則直接讀快取,避免重跑 MMPose。

    get_inferencer: 選用的 callable,回傳一個共用的 MMPoseInferencer。批量模式
        傳入此參數,讓整批影片共用同一個 inferencer(只載入一次模型),避免每支
        影片各自建立 inferencer 累積佔用 GPU 記憶體而 CUDA OOM。
    """
    if cache_path and Path(cache_path).exists():
        print(f"[cache] 讀取既有骨架快取: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    if get_inferencer is not None:
        inferencer = get_inferencer()
    else:
        from mmpose.apis import MMPoseInferencer
        print("[mmpose] 載入 inferencer 'human' ...")
        inferencer = MMPoseInferencer('human')

    frames = []
    print("[mmpose] 逐幀抽取骨架(第一次會比較久)...")
    for result in tqdm(inferencer(video_path, show=False)):
        preds = result['predictions'][0]
        if len(preds) == 0:
            frames.append({
                'kps': np.zeros((0, 17, 2), np.float32),
                'scores': np.zeros((0, 17), np.float32),
                'bboxes': np.zeros((0, 4), np.float32),
            })
            continue
        kps = np.array([p['keypoints'] for p in preds], np.float32)      # (n,17,2)
        scr = np.array([p['keypoint_scores'] for p in preds], np.float32)  # (n,17)
        bbs = np.array([p['bbox'][0] for p in preds], np.float32)          # (n,4)
        frames.append({'kps': kps, 'scores': scr, 'bboxes': bbs})

    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump(frames, f)
        print(f"[cache] 骨架已快取到: {cache_path}")
    return frames


# ====================================================================
# Step 2: (選用)用球場 homography 選出場內 Top/Bottom 兩名球員
# ====================================================================

def _transform_points(H, points):
    """用 3x3 homography 轉換 Nx2 點，回傳 Nx2。"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"points 必須是 Nx2，實際 shape={pts.shape}")

    pts_h = np.column_stack([pts, np.ones(len(pts), dtype=np.float64)])
    mapped_h = (np.asarray(H, dtype=np.float64) @ pts_h.T).T
    denom = mapped_h[:, 2]

    if np.any(np.isclose(denom, 0.0)):
        raise ValueError("homography 轉換得到無效齊次座標（分母接近 0）")

    return mapped_h[:, :2] / denom[:, None]


def _rectangle_score(points):
    """估計四點轉換後有多接近矩形；分數越小越好。"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape != (4, 2) or not np.isfinite(pts).all():
        return float("inf")

    edges = np.roll(pts, -1, axis=0) - pts
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths < 1e-9):
        return float("inf")

    # 相鄰邊應接近垂直。
    orthogonality = np.mean([
        abs(np.dot(edges[i], edges[(i + 1) % 4]) /
            (lengths[i] * lengths[(i + 1) % 4]))
        for i in range(4)
    ])

    # 對邊長度應接近。
    opposite_length_error = (
        abs(lengths[0] - lengths[2]) / (lengths[0] + lengths[2]) +
        abs(lengths[1] - lengths[3]) / (lengths[1] + lengths[3])
    )

    # 軸對齊矩形的面積應接近 bounding box 面積。
    bbox_area = np.ptp(pts[:, 0]) * np.ptp(pts[:, 1])
    polygon_area = 0.5 * abs(
        np.dot(pts[:, 0], np.roll(pts[:, 1], 1)) -
        np.dot(pts[:, 1], np.roll(pts[:, 0], 1))
    )
    fill_error = (
        abs(1.0 - polygon_area / bbox_area)
        if bbox_area > 1e-9 else float("inf")
    )

    return float(orthogonality + opposite_length_error + fill_error)


def load_court_homography_any(path: str):
    """讀取舊版球場 CSV 或主系統輸出的 court.json。

    回傳:
        H_image_to_court:
            影像像素座標 -> 球場平面座標的 3x3 homography。
        court_bounds:
            select_in_court() 使用的球場平面邊界。

    court.json 預期格式:
        {
          "courts": [{
            "corners": [[x, y], ... 4 points],
            "homography": [[...], [...], [...]],
            "segment_index": null
          }]
        }

    部分 court_detection 輸出的 homography 是 court -> image，部分舊契約
    則描述為 image -> court。此 loader 會用 corners 自動判斷方向。
    """
    p = Path(path)

    if p.suffix.lower() != ".json":
        from court_to_homography import load_court_homography
        return load_court_homography(path)

    with p.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    courts = payload.get("courts")
    if not isinstance(courts, list) or not courts:
        raise ValueError(f"{p} 缺少非空的 'courts' 陣列")

    # 優先使用全場共用的 court（segment_index=null）；若沒有則取第一筆。
    court = next(
        (item for item in courts if item.get("segment_index") is None),
        courts[0],
    )

    corners = np.asarray(court.get("corners"), dtype=np.float64)
    H_raw = np.asarray(court.get("homography"), dtype=np.float64)

    if corners.shape != (4, 2):
        raise ValueError(
            f"{p} 的 courts[].corners 必須是 4x2，實際 shape={corners.shape}"
        )
    if H_raw.shape != (3, 3):
        raise ValueError(
            f"{p} 的 courts[].homography 必須是 3x3，實際 shape={H_raw.shape}"
        )
    if not np.isfinite(corners).all() or not np.isfinite(H_raw).all():
        raise ValueError(f"{p} 的 corners/homography 含有 NaN 或 Infinity")

    try:
        H_inverse = np.linalg.inv(H_raw)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{p} 的 homography 不可逆") from exc

    # 兩種可能:
    #   1. H_raw 已是 image -> court
    #   2. H_raw 是 court -> image，需反矩陣
    mapped_by_raw = _transform_points(H_raw, corners)
    mapped_by_inverse = _transform_points(H_inverse, corners)

    raw_score = _rectangle_score(mapped_by_raw)
    inverse_score = _rectangle_score(mapped_by_inverse)

    if inverse_score < raw_score:
        H_image_to_court = H_inverse
        court_plane_corners = mapped_by_inverse
        direction = "court->image（已自動反矩陣）"
    else:
        H_image_to_court = H_raw
        court_plane_corners = mapped_by_raw
        direction = "image->court"

    x_values = court_plane_corners[:, 0]
    y_values = court_plane_corners[:, 1]
    court_bounds = {
        "border_L": float(x_values.min()),
        "border_R": float(x_values.max()),
        "border_U": float(y_values.min()),
        "border_D": float(y_values.max()),
    }

    if np.isclose(court_bounds["border_L"], court_bounds["border_R"]) or \
       np.isclose(court_bounds["border_U"], court_bounds["border_D"]):
        raise ValueError(f"{p} 的球場邊界退化，無法用於選人")

    selected_segment = court.get("segment_index")
    print(
        "[court] JSON 欄位載入完成: "
        f"segment_index={selected_segment}, "
        f"homography={direction}, "
        f"bounds=({court_bounds['border_L']:.3f}, "
        f"{court_bounds['border_R']:.3f}, "
        f"{court_bounds['border_U']:.3f}, "
        f"{court_bounds['border_D']:.3f})"
    )

    return H_image_to_court, court_bounds


def select_in_court(frame_det, H, court_bounds, eps=0.01, y_margin=0.12):
    """
    比照 bst_infer_from_video.extract_per_frame_features 的選人邏輯。
    回傳 (top_idx, bottom_idx);選不到回 (None, None)。
    idx 是 frame_det['kps'] 內的列索引。

    y_margin: 上下(底線方向)額外放寬的比例。起跳殺球時球員雙腳離地,
        homography(假設腳在地面)會把投影落點推到底線之外,使 yn < 0(Top)
        或 yn > 1(Bottom)而被誤排除。把上下邊界各往外放寬 y_margin(底線
        方向長度的比例)就能把起跳的球員撈回來。左右(邊線方向)仍只用很小的
        eps,避免把坐在邊線外的線審 / 裁判也抓進來。
    """
    kps = frame_det['kps']
    if len(kps) < 2:
        return None, None

    feet = kps[:, -2:, :]            # (n, 2, 2) 兩腳踝
    feet_mid = feet.mean(axis=1)     # (n, 2)
    feet_cam = feet_mid.T            # (2, n)
    feet_cam_h = np.vstack([feet_cam, np.ones((1, feet_cam.shape[1]))])
    feet_court = H @ feet_cam_h
    feet_court = feet_court[:2] / feet_court[2:]   # (2, n)

    xL, xR = court_bounds['border_L'], court_bounds['border_R']
    yU, yD = court_bounds['border_U'], court_bounds['border_D']
    xn = (feet_court[0] - xL) / (xR - xL)
    yn = (feet_court[1] - yU) / (yD - yU)

    in_court = (xn > -eps) & (xn < 1 + eps) & \
               (yn > -y_margin) & (yn < 1 + y_margin)
    ids = np.nonzero(in_court)[0]
    if len(ids) < 2:
        return None, None

    pid = ids[:2]
    # y 較小的是 Top
    if yn[pid[0]] > yn[pid[1]]:
        pid = pid[::-1]
    return int(pid[0]), int(pid[1])


# ====================================================================
# Step 3: (選用)讀球軌跡 / 擊球影格
# ====================================================================
def load_shuttle(csv_path: str):
    import pandas as pd
    df = pd.read_csv(csv_path)
    if 'X' in df.columns:
        coords = df[['X', 'Y']].to_numpy(float)
    else:
        coords = df[['CX', 'CY']].to_numpy(float)
    if 'Visibility' in df.columns:
        coords[df['Visibility'].to_numpy() == 0] = 0.0
    return coords


def load_hits(path: str):
    """回傳擊球 frame 的 set。沿用 bst_infer_from_video 的容錯讀檔。"""
    p = Path(path)
    encs = ('utf-8-sig', 'utf-16', 'utf-8', 'cp950', 'latin-1')
    if p.suffix.lower() == '.csv':
        import pandas as pd
        df = None
        for enc in encs:
            try:
                df = pd.read_csv(p, encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        col = next((c for c in ['frame_num', 'HitFrame', 'hit_frame', 'frame', 'Frame']
                    if c in df.columns), None)
        if col is None:
            raise ValueError(f"hit CSV 找不到 frame 欄位,欄位有: {list(df.columns)}")
        return set(df[col].astype(int).tolist())
    else:
        text = None
        for enc in encs:
            try:
                text = p.read_text(encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        return {int(float(x)) for x in text.split() if x.strip()}


# ====================================================================
# Step 4: 畫骨架
# ====================================================================
COLOR = {'l': (80, 160, 255), 'r': (255, 180, 80), 'c': (230, 230, 230)}  # BGR


def draw_one_person(img, kps, scores, kp_thr, line_color, thickness=2,
                    joint_color=None, dim=False):
    """畫單一個人的骨架。dim=True 時用灰色細線(沒被選中的人)。"""
    if dim:
        line_color = (130, 130, 130)
        thickness = 1

    for a, b, side in BONES:
        if scores[a] < kp_thr or scores[b] < kp_thr:
            continue
        pa = tuple(np.round(kps[a]).astype(int))
        pb = tuple(np.round(kps[b]).astype(int))
        c = line_color if (dim or line_color is not None) else COLOR[side]
        cv2.line(img, pa, pb, c, thickness, cv2.LINE_AA)

    for j in range(17):
        p = tuple(np.round(kps[j]).astype(int))
        if scores[j] < kp_thr:
            # 低信心關節 → 紅色小點
            cv2.circle(img, p, 2, (40, 40, 220), -1, cv2.LINE_AA)
            continue
        jc = joint_color if joint_color is not None else (255, 255, 255)
        r = 3
        if not dim and j in WRIST_IDS:
            # 手腕加大,擊球關鍵點
            cv2.circle(img, p, 7, (0, 255, 255), 2, cv2.LINE_AA)
            r = 4
        cv2.circle(img, p, r, jc, -1, cv2.LINE_AA)


def render_frame(img, frame_det, kp_thr, top_idx=None, bottom_idx=None,
                 shuttle_xy=None, is_hit=False, frame_idx=None, draw_bbox=True):
    """在 img(會被就地修改)上畫出該幀所有標註。"""
    kps_all = frame_det['kps']
    scr_all = frame_det['scores']
    bb_all = frame_det['bboxes']
    n = len(kps_all)

    selected = {top_idx, bottom_idx} - {None}

    # 先畫沒被選中的人(灰),再畫被選中的(亮),確保亮色在上層
    for i in range(n):
        if i in selected:
            continue
        draw_one_person(img, kps_all[i], scr_all[i], kp_thr, None, dim=True)

    def draw_selected(idx, color, label):
        if idx is None:
            return
        draw_one_person(img, kps_all[idx], scr_all[idx], kp_thr,
                        line_color=color, thickness=2, joint_color=(255, 255, 255))
        if draw_bbox and idx < len(bb_all):
            x1, y1, x2, y2 = np.round(bb_all[idx]).astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
            cv2.putText(img, label, (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    draw_selected(top_idx, (80, 230, 80), 'TOP')        # 綠
    draw_selected(bottom_idx, (250, 160, 60), 'BOTTOM')  # 藍

    # 球
    if shuttle_xy is not None and not (shuttle_xy[0] == 0 and shuttle_xy[1] == 0):
        sp = tuple(np.round(shuttle_xy).astype(int))
        cv2.circle(img, sp, 6, (40, 40, 230), -1, cv2.LINE_AA)
        cv2.circle(img, sp, 9, (255, 255, 255), 1, cv2.LINE_AA)

    # 左上資訊列
    info = f"frame {frame_idx}  det={n}"
    cv2.rectangle(img, (0, 0), (260, 26), (0, 0, 0), -1)
    cv2.putText(img, info, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)

    # 擊球橫幅
    if is_hit:
        h, w = img.shape[:2]
        cv2.rectangle(img, (0, 0), (w, 6), (0, 0, 255), -1)
        cv2.putText(img, 'HIT', (w - 80, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2, cv2.LINE_AA)
    return img


# ====================================================================
# Step 5: 主流程
# ====================================================================
def build_context(args, court_data=None, get_inferencer=None):
    """準備所有逐幀標註所需資料,回傳一個 dict。

    court_data: (H, court_bounds);批量模式預先載入一次共用,避免每支影片重讀。
    get_inferencer: 批量模式共用的 MMPose inferencer getter(見 extract_skeletons)。
    """
    dets = extract_skeletons(args.video, args.cache, get_inferencer=get_inferencer)
    n_frames = len(dets)

    H = court_bounds = None
    if court_data is not None:
        H, court_bounds = court_data
    elif args.court:
        H, court_bounds = load_court_homography_any(args.court)
        print(f"[court] 已載入球場 homography: {args.court}")

    shuttle = None
    if args.shuttle:
        shuttle = load_shuttle(args.shuttle)
        print(f"[shuttle] 已載入球軌跡: {args.shuttle}  ({len(shuttle)} 點)")

    hits = set()
    if args.hits:
        hits = load_hits(args.hits)
        print(f"[hits] 已載入 {len(hits)} 個擊球影格")

    # 預先算每幀的 Top/Bottom 選擇(若有 court)
    sel = [(None, None)] * n_frames
    if H is not None:
        for i, d in enumerate(dets):
            sel[i] = select_in_court(d, H, court_bounds, y_margin=args.y_margin)

    return {
        'dets': dets, 'n_frames': n_frames, 'sel': sel,
        'shuttle': shuttle, 'hits': hits, 'kp_thr': args.kp_thr,
    }


def get_annotation(ctx, i, img):
    """取得第 i 幀的標註並畫到 img 上。"""
    det = ctx['dets'][i]
    top_idx, bottom_idx = ctx['sel'][i]
    sx = ctx['shuttle'][i] if (ctx['shuttle'] is not None and i < len(ctx['shuttle'])) else None
    return render_frame(
        img, det, ctx['kp_thr'],
        top_idx=top_idx, bottom_idx=bottom_idx,
        shuttle_xy=sx, is_hit=(i in ctx['hits']), frame_idx=i,
    )


def run_save(args, ctx):
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

    n = min(ctx['n_frames'], int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or ctx['n_frames'])
    print(f"[save] 輸出標註影片到 {args.save} ({w}x{h} @ {fps:.1f}fps, {n} 幀)")
    for i in tqdm(range(n)):
        ok, img = cap.read()
        if not ok:
            break
        get_annotation(ctx, i, img)
        writer.write(img)
    cap.release()
    writer.release()
    print(f"[save] 完成: {args.save}")


def run_show(args, ctx):
    cap = cv2.VideoCapture(args.video)
    n = ctx['n_frames']
    hits_sorted = sorted(ctx['hits'])

    win = 'skeleton check  (space=play  arrows=step  n/p=hit  s=save  Esc=quit)'
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    i = 0
    playing = False
    print("[show] 開啟互動視窗。空白=播放/暫停, ←→=逐幀, n/p=擊球幀, s=存圖, Esc=離開")

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        return img if ok else None

    while True:
        i = max(0, min(n - 1, i))
        img = read_frame(i)
        if img is None:
            break
        get_annotation(ctx, i, img)
        cv2.imshow(win, img)

        key = cv2.waitKey(int(1000 / 30) if playing else 0) & 0xFF
        if key == 27:                      # Esc
            break
        elif key == ord(' '):
            playing = not playing
        elif key in (ord('d'), 83):        # → next
            i += 1; playing = False
        elif key in (ord('a'), 81):        # ← prev
            i -= 1; playing = False
        elif key in (ord('e'), ord('.')):
            i += 10; playing = False
        elif key in (ord('q'), ord(',')):
            i -= 10; playing = False
        elif key == ord('n'):              # 下一個擊球幀
            nxt = [h for h in hits_sorted if h > i]
            if nxt:
                i = nxt[0]; playing = False
        elif key == ord('p'):              # 上一個擊球幀
            prv = [h for h in hits_sorted if h < i]
            if prv:
                i = prv[-1]; playing = False
        elif key == ord('s'):
            out = f"frame_{i:06d}.png"
            cv2.imwrite(out, img)
            print(f"[show] 已存 {out}")

        if playing:
            i += 1
            if i >= n:
                playing = False
                i = n - 1

    cap.release()
    cv2.destroyAllWindows()


def run_save_csv(args, ctx):
    """
    輸出每幀場內 Top/Bottom 兩名球員的 COCO-17 骨架(原始像素座標)成 CSV。
    每幀固定兩列 (player=top / player=bottom);該幀選不到該位球員時,
    座標與信心欄留空,det_idx 留空,方便對齊 frame 索引。
    欄位:
        frame, player, det_idx,
        bbox_x1, bbox_y1, bbox_x2, bbox_y2,
        <joint>_x, <joint>_y, <joint>_s   (17 個關節)
    """
    import csv

    header = ['frame', 'player', 'det_idx',
              'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2']
    for name in COCO_NAMES:
        header += [f'{name}_x', f'{name}_y', f'{name}_s']

    n = ctx['n_frames']
    print(f"[save-csv] 輸出 Top/Bottom 骨架到 {args.save_csv} ({n} 幀 × 2 列)")

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

    with open(args.save_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        n_top = n_bottom = 0
        for i in range(n):
            det = ctx['dets'][i]
            top_idx, bottom_idx = ctx['sel'][i]
            n_top += top_idx is not None
            n_bottom += bottom_idx is not None
            writer.writerow(player_row(i, 'top', top_idx, det))
            writer.writerow(player_row(i, 'bottom', bottom_idx, det))

    print(f"[save-csv] 完成: {args.save_csv}  "
          f"(top 有效 {n_top}/{n} 幀, bottom 有效 {n_bottom}/{n} 幀)")


VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.MP4', '.AVI', '.MOV', '.MKV')


def _match_by_stem(stem: str, dir_path: Path, exts):
    """在 dir_path 內找檔名(去副檔名)等於 stem 的檔案,找不到回 None。"""
    for ext in exts:
        cand = dir_path / f'{stem}{ext}'
        if cand.exists():
            return cand
    return None


def run_batch(args):
    """
    批量處理一個資料夾內的所有影片(用法一:標註影片 / 用法四:骨架 CSV)。
    全部影片共用同一個 --court(只讀一次);shuttle / hits 若提供資料夾則依檔名配對。
    """
    from argparse import Namespace

    video_dir = Path(args.video_dir)
    if not video_dir.is_dir():
        raise NotADirectoryError(f"--video-dir 不是資料夾: {video_dir}")

    videos = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix in VIDEO_EXTS
    )
    if not videos:
        raise FileNotFoundError(f"{video_dir} 內找不到任何影片 ({', '.join(set(VIDEO_EXTS))})")

    out_dir = Path(args.out_dir) if args.out_dir else video_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 場地 homography 只讀一次,全部影片共用
    court_data = None
    if args.court:
        court_data = load_court_homography_any(args.court)
        print(f"[court] 已載入共用球場 homography: {args.court}")

    shuttle_dir = Path(args.shuttle_dir) if args.shuttle_dir else None
    hits_dir = Path(args.hits_dir) if args.hits_dir else None

    # MMPose inferencer 整批只載入一次共用,避免每支影片各自建立而累積 GPU 記憶體
    # 造成 CUDA out of memory。用 lazy getter,全部影片都有快取時就完全不載入模型。
    _inf: dict = {'obj': None}

    def get_inferencer():
        if _inf['obj'] is None:
            from mmpose.apis import MMPoseInferencer
            print("[mmpose] 載入 inferencer 'human' (整批共用,只載入一次) ...")
            _inf['obj'] = MMPoseInferencer('human')
        return _inf['obj']

    print(f"[batch] 共 {len(videos)} 支影片,輸出到 {out_dir}")
    for n, video in enumerate(videos, 1):
        stem = video.stem
        print(f"\n[batch] ({n}/{len(videos)}) {video.name}")

        # 每支影片各自的 shuttle / hits(依檔名配對,找不到就略過)
        shuttle = None
        if shuttle_dir is not None:
            m = _match_by_stem(stem, shuttle_dir, ('.csv',))
            shuttle = str(m) if m else None
        hits = None
        if hits_dir is not None:
            m = _match_by_stem(stem, hits_dir, ('.csv', '.txt'))
            hits = str(m) if m else None

        sub = Namespace(
            video=str(video),
            court=args.court,
            shuttle=shuttle,
            hits=hits,
            kp_thr=args.kp_thr,
            y_margin=args.y_margin,
            cache=str(video.with_suffix('')) + '_pose.pkl',
            save=str(out_dir / f'{stem}_skeleton.mp4') if args.save else None,
            save_csv=str(out_dir / f'{stem}_skeleton.csv') if args.save_csv else None,
            show=False,
        )

        ctx = build_context(sub, court_data=court_data, get_inferencer=get_inferencer)
        if sub.save_csv:
            run_save_csv(sub, ctx)
        if sub.save:
            run_save(sub, ctx)

    print(f"\n[batch] 全部完成,共處理 {len(videos)} 支影片。")


def main():
    ap = argparse.ArgumentParser(description="視覺化檢查 MMPose 骨架標記是否準確")
    ap.add_argument('--video', help='輸入影片 (單支模式)')
    ap.add_argument('--video-dir', dest='video_dir',
                    help='批量模式:輸入影片資料夾,逐支處理 (用法一 / 用法四)')
    ap.add_argument('--out-dir', dest='out_dir',
                    help='批量模式輸出資料夾 (預設與 --video-dir 相同);'
                         '檔名為 <影片名>_skeleton.mp4 / <影片名>_skeleton.csv')
    ap.add_argument('--save', nargs='?', const=True, default=None,
                    help='輸出標註影片;單支模式請給路徑 (例 out.mp4),批量模式當旗標用')
    ap.add_argument('--save-csv', dest='save_csv', nargs='?', const=True, default=None,
                    help='輸出 Top/Bottom 兩名球員骨架 CSV (原始像素座標);需要 --court。'
                         '單支模式給路徑,批量模式當旗標用')
    ap.add_argument('--show', action='store_true', help='開啟互動逐幀檢視視窗 (僅單支模式)')
    ap.add_argument('--court', help='球場資料 court.json 或舊版球場 CSV;給了才會標 Top/Bottom')
    ap.add_argument('--shuttle', help='TrackNet 球軌跡 CSV (單支模式)')
    ap.add_argument('--hits', help='擊球影格清單 csv/txt (單支模式)')
    ap.add_argument('--shuttle-dir', dest='shuttle_dir',
                    help='批量模式:球軌跡 CSV 資料夾,依影片檔名配對')
    ap.add_argument('--hits-dir', dest='hits_dir',
                    help='批量模式:擊球影格資料夾,依影片檔名配對')
    ap.add_argument('--kp-thr', type=float, default=0.3,
                    help='關節信心門檻,低於此值畫成紅點 (預設 0.3)')
    ap.add_argument('--y-margin', dest='y_margin', type=float, default=0.12,
                    help='選人時上下(底線方向)額外放寬的比例,讓起跳殺球的球員不被'
                         '排除;太大會抓到底線外的裁判 (預設 0.12)')
    ap.add_argument('--cache', help='骨架快取 .pkl 路徑 (預設 <video>_pose.pkl)')
    args = ap.parse_args()

    # ---------------- 批量模式 ----------------
    if args.video_dir:
        if args.video:
            ap.error("--video 與 --video-dir 不可同時使用。")
        if args.show:
            ap.error("--show 僅支援單支模式,批量模式請用 --save / --save-csv。")
        if not args.save and not args.save_csv:
            ap.error("批量模式請至少指定 --save (用法一) 或 --save-csv (用法四)。")
        if args.save_csv and not args.court:
            ap.error("--save-csv 只輸出場內 Top/Bottom 兩名球員,必須同時提供 --court。")
        run_batch(args)
        return

    # ---------------- 單支模式 ----------------
    if not args.video:
        ap.error("請提供 --video (單支) 或 --video-dir (批量)。")

    if not args.save and not args.show and not args.save_csv:
        ap.error("請至少指定 --save / --save-csv / --show 其中之一。")

    if args.save is True or args.save_csv is True:
        ap.error("單支模式 --save / --save-csv 需要給輸出路徑 (旗標用法僅限批量模式)。")

    if args.save_csv and not args.court:
        ap.error("--save-csv 只輸出場內 Top/Bottom 兩名球員,必須同時提供 --court。")

    if args.cache is None:
        args.cache = str(Path(args.video).with_suffix('')) + '_pose.pkl'

    ctx = build_context(args)

    if args.save_csv:
        run_save_csv(args, ctx)
    if args.save:
        run_save(args, ctx)
    if args.show:
        run_show(args, ctx)


if __name__ == '__main__':
    main()
