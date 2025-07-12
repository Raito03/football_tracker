
import os
import sys
import cv2
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import torch
import numpy as np

def load_model(path):
    # 1. Basic file checks
    if not os.path.isfile(path):
        print(f"Error: model file not found at {path}")
        sys.exit(1)
    if os.path.getsize(path) == 0:
        print(f"Error: model file at {path} is empty")
        sys.exit(1)

    # 2. Allowlisting DetectionModel for safe unpickle and load on GPU
    torch.serialization.add_safe_globals([DetectionModel])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    try:
        torch.load(path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Error loading with torch.load: {e}")
        sys.exit(1)

    # 3. Load with Ultralytics YOLO
    try:
        model = YOLO(path)
        model.to(device)
        print("Model classes:", model.names)
        return model
    except Exception as e:
        print(f"Error initializing YOLO model: {e}")
        sys.exit(1)


def draw_player_annotations(frame, player_box, ball_centers, proximity_thresh=30):
    x1, y1, x2, y2 = player_box
    feet_x = (x1 + x2) // 2
    feet_y = y2

    color = (255, 255, 0)  # default: cyan
    if ball_centers:
        distances = [np.hypot(feet_x - bx, feet_y - by) for (bx, by) in ball_centers]
        if any(dist <= proximity_thresh for dist in distances):
            color = (0, 255, 0)  # green if ball is close

    # Tactical cam ellipse parameters
    axes = ((x2 - x1) // 3, 4)
    angle = -3
    cv2.ellipse(frame, (feet_x, feet_y), axes, angle, 0, 360, color, 2)

    return color


def draw_ball_tracker(frame, ball_center):
    bx, by = ball_center
    # Define a downward-pointing triangle above the ball
    size = 8  # triangle half-width
    height = 16  # triangle height
    pts = np.array([
        [bx - size, by - height],  # left corner
        [bx + size, by - height],  # right corner
        [bx, by]                   # bottom tip at ball center
    ], np.int32)
    pts = pts.reshape((-1, 1, 2))
    # Fill the triangle with green
    cv2.fillPoly(frame, [pts], (0, 255, 0))



def main():
    model_path = 'best.pt'
    model = load_model(model_path)

    input_path = 'tacticam.mp4'
    output_path = 'tacticam_tracked_bytetrack.mp4'

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: could not open video {input_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    results = model.track(
        source=input_path,
        stream=True,
        tracker='bytetrack.yaml',
        persist=True,
        hide_labels=False,  # show labels for debugging
        hide_conf=False,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',
    )

    frame_idx = 0
    for result in results:
        frame = result.orig_img
        ball_centers = []

        # Collect ball centers
        for box in result.boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = model.names.get(cls_id, str(cls_id)).lower()
            if 'ball' in class_name:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                ball_centers.append((cx, cy))
                draw_ball_tracker(frame, (cx, cy))

        # Debug: if no ball detected
        if not ball_centers:
            print(f"Frame {frame_idx}: No ball detected (classes on frame: ",
                  [model.names[int(b.cls[0])] for b in result.boxes], ")")

        # Draw player annotations
        for box in result.boxes:
            cls_id = int(box.cls[0].cpu().numpy())
            class_name = model.names.get(cls_id, str(cls_id)).lower()
            if 'ball' not in class_name:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                track_id = int(box.id[0].cpu().numpy())
                color = draw_player_annotations(frame, (x1, y1, x2, y2), ball_centers)
                head_x = (x1 + x2) // 2
                cv2.putText(frame, f"ID:{track_id}", (head_x, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    print(f"Done! Output saved to {output_path}")


if __name__ == '__main__':
    main()
