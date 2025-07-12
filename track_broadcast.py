
import os
import sys
import cv2
import torch
import torch.nn.functional as F
from ultralytics import YOLO
import numpy as np
import torchvision.transforms as T
import torchvision.models as models
from scipy.optimize import linear_sum_assignment
import pickle
from collections import defaultdict


def load_model(path):
    if not os.path.isfile(path) or os.path.getsize(path)==0:
        print(f"Invalid model file at {path}")
        sys.exit(1)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    try:
        model = YOLO(path)
        model.to(device)
        return model, device
    except Exception as e:
        print(f"Error initializing YOLO: {e}")
        sys.exit(1)


def init_reid_model(device):
    reid = models.resnet50(pretrained=True)
    reid.fc = torch.nn.Identity()
    reid.to(device).eval()

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return reid, transform


def extract_embedding(crop, reid, transform, device):
    if crop.size == 0:
        return torch.zeros(2048).to(device)

    try:
        x = transform(crop).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = reid(x)
        return F.normalize(feat, p=2, dim=1).squeeze(0)
    except Exception:
        return torch.zeros(2048).to(device)


def draw_player_annotations(frame, player_box, ball_centers, track_id=None, proximity_thresh=30):
    x1, y1, x2, y2 = player_box
    feet_x = (x1 + x2) // 2
    feet_y = y2

    color = (255, 255, 0)
    if ball_centers:
        distances = [np.hypot(feet_x - bx, feet_y - by) for (bx, by) in ball_centers]
        if any(dist <= proximity_thresh for dist in distances):
            color = (0, 255, 0)

    axes = ((x2 - x1) // 3, 4)
    angle = -3
    cv2.ellipse(frame, (feet_x, feet_y), axes, angle, 0, 360, color, 2)

    if track_id is not None:
        head_x = (x1 + x2) // 2
        cv2.putText(frame, f"ID:{track_id}", (head_x, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)


def draw_ball_tracker(frame, ball_center):
    bx, by = ball_center
    size = 8
    height = 16
    pts = np.array([
        [bx - size, by - height],
        [bx + size, by - height],
        [bx, by]
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(frame, [pts], (0, 255, 0))


def track_and_embed(video_path, model, reid, transform, device):
    tact_embs = defaultdict(list)
    cap = cv2.VideoCapture(video_path)

    results = model.track(source=video_path, stream=True, tracker='bytetrack.yaml', persist=True,
                          hide_labels=True, hide_conf=True, device=device)

    for res in results:
        frame = res.orig_img
        for box in res.boxes:
            cls_id = int(box.cls[0].cpu())
            name = model.names[cls_id].lower()
            if 'ball' in name:
                continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            track_id = int(box.id[0].cpu())
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] == 0 or crop.shape[1] == 0:
                continue
            emb = extract_embedding(crop, reid, transform, device)
            tact_embs[track_id].append(emb)

    cap.release()
    return tact_embs


def compute_mean_embeddings(tact_embs):
    mean_embs = {}
    for pid, embs in tact_embs.items():
        if embs:
            mean_embs[pid] = F.normalize(torch.stack(embs).mean(0), p=2, dim=0)
    return mean_embs


def match_embeddings_hungarian(curr_embs, mean_embs):
    if not mean_embs or not curr_embs:
        return {}
    pids = list(mean_embs.keys())
    cost = np.zeros((len(curr_embs), len(pids)), dtype=np.float32)
    for i, emb in enumerate(curr_embs):
        for j, pid in enumerate(pids):
            sim = F.cosine_similarity(emb.unsqueeze(0), mean_embs[pid].unsqueeze(0))
            cost[i, j] = -sim.item()
    row_idx, col_idx = linear_sum_assignment(cost)
    return {i: pids[j] if -cost[i, j] > 0.3 else None for i, j in zip(row_idx, col_idx)}


def temporal_smoothing(assignments, max_gap=10):
    smoothed = {}
    last_valid = {}
    for frame_idx in sorted(assignments):
        frame_ids = assignments[frame_idx]
        smooth_frame = []
        for i, pid in enumerate(frame_ids):
            if pid is not None:
                last_valid[i] = (frame_idx, pid)
                smooth_frame.append(pid)
            else:
                if i in last_valid and frame_idx - last_valid[i][0] <= max_gap:
                    smooth_frame.append(last_valid[i][1])
                else:
                    smooth_frame.append(None)
        smoothed[frame_idx] = smooth_frame
    return smoothed


def visualize_broadcast(video_path, output_path, model, reid, transform, device, mean_embs):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    assignments = {}
    results = model.track(source=video_path, stream=True, tracker='bytetrack.yaml', persist=True,
                          hide_labels=True, hide_conf=True, device=device)

    frame_idx = 0
    for res in results:
        frame = res.orig_img
        current_embs = []
        bboxes = []
        for box in res.boxes:
            cls_id = int(box.cls[0].cpu())
            if 'ball' in model.names[cls_id].lower(): continue
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] == 0 or crop.shape[1] == 0: continue
            emb = extract_embedding(crop, reid, transform, device)
            current_embs.append(emb)
            bboxes.append((x1, y1, x2, y2))
        matches = match_embeddings_hungarian(current_embs, mean_embs)
        assignments[frame_idx] = [matches.get(i) for i in range(len(current_embs))]
        for i, (x1, y1, x2, y2) in enumerate(bboxes):
            draw_player_annotations(frame, (x1, y1, x2, y2), [], track_id=assignments[frame_idx][i])
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    smoothed = temporal_smoothing(assignments)

    print(f"Saved results to {output_path}")
    return smoothed


def save_embeddings(tact_embs, path='tactic_embeddings.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(dict(tact_embs), f)
    print(f"Saved embeddings to {path}")


def load_embeddings(path='tactic_embeddings.pkl'):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, _ = load_model('best.pt')
    reid, transform = init_reid_model(device)

    if os.path.exists('tactic_embeddings.pkl'):
        tact_embs = load_embeddings()
    else:
        tact_embs = track_and_embed('tacticam.mp4', model, reid, transform, device)
        save_embeddings(tact_embs)

    mean_embs = compute_mean_embeddings(tact_embs)
    visualize_broadcast('broadcast.mp4', 'broadcast_tracked.mp4',
                        model, reid, transform, device, mean_embs)


if __name__ == '__main__':
    main()
