import argparse
import av
import cv2
import mediapipe as mp
import os
import json
import time
import numpy as np
import torch
from scipy.interpolate import interp1d
from ultralytics import YOLO

def serialize_landmarks(landmark_list):
    if not landmark_list:
        return []
    return [
        {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z), "visibility": float(lm.visibility) if hasattr(lm, "visibility") else 1.0}
        for lm in landmark_list.landmark
    ]

class KalmanHandTracker:
    def __init__(self, num_points=21):
        self.num_points = num_points
        self.state = np.zeros((num_points, 6))
        self.P = np.array([np.eye(6) * 1000.0 for _ in range(num_points)])
        self.F = np.eye(6)
        self.F[0, 3] = 1; self.F[1, 4] = 1; self.F[2, 5] = 1
        self.H = np.zeros((3, 6))
        self.H[0, 0] = 1; self.H[1, 1] = 1; self.H[2, 2] = 1
        self.R = np.eye(3) * 10.0
        self.Q = np.eye(6) * 1.0
        self.frames_since_update = 0
        self.is_initialized = False

    def predict(self):
        if not self.is_initialized: return None
        for i in range(self.num_points):
            self.state[i] = self.F @ self.state[i]
            # Apply velocity dampening (friction) so points don't fly off screen
            self.state[i, 3:] *= 0.6
            self.P[i] = self.F @ self.P[i] @ self.F.T + self.Q
        self.frames_since_update += 1
        return [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2]), "visibility": 0.5} for p in self.state]

    def update(self, landmarks):
        if not landmarks: return
        pts = np.array([[lm['x'], lm['y'], lm['z']] for lm in landmarks])
        if not self.is_initialized:
            for i in range(self.num_points):
                self.state[i, :3] = pts[i]
            self.is_initialized = True
        else:
            for i in range(self.num_points):
                y = pts[i] - self.H @ self.state[i]
                S = self.H @ self.P[i] @ self.H.T + self.R
                K = self.P[i] @ self.H.T @ np.linalg.inv(S)
                self.state[i] = self.state[i] + K @ y
                self.P[i] = (np.eye(6) - K @ self.H) @ self.P[i]
        self.frames_since_update = 0

def crop_and_detect_hand(image_rgb, wrist_x, wrist_y, hands_model, target_label, elbow_x=None, elbow_y=None):
    h, w, _ = image_rgb.shape
    
    # Conditional Kinematics
    if elbow_x is not None and elbow_y is not None:
        vx = wrist_x - elbow_x
        vy = wrist_y - elbow_y
        forearm_len = np.sqrt(vx**2 + vy**2)
        cx = int((wrist_x + vx * 0.3) * w)
        cy = int((wrist_y + vy * 0.3) * h)
        primary_box_size = int(max(h, w) * max(0.2, min(0.6, forearm_len * 1.5)))
        scales = [primary_box_size, int(primary_box_size * 0.7), int(primary_box_size * 1.3)]
    else:
        cx, cy = int(wrist_x * w), int(wrist_y * h)
        primary_box_size = int(max(h, w) * 0.4) # Slightly larger fallback
        scales = [primary_box_size, int(max(h, w) * 0.25), int(max(h, w) * 0.55)]
        
    for box_size in scales:
        x1, y1 = max(0, cx - box_size // 2), max(0, cy - box_size // 2)
        x2, y2 = min(w, cx + box_size // 2), min(h, cy + box_size // 2)
        
        if x2 <= x1 or y2 <= y1:
            continue
            
        crop = image_rgb[y1:y2, x1:x2]
        results = hands_model.process(crop)
    
    if results.multi_hand_landmarks and results.multi_handedness:
        # Search for the correct handedness
        for hand_lms, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if handedness.classification[0].label == target_label:
                adjusted_lms = []
                for lm in hand_lms.landmark:
                    abs_x = x1 + lm.x * (x2 - x1)
                    abs_y = y1 + lm.y * (y2 - y1)
                    adjusted_lms.append({
                        "x": abs_x / w, "y": abs_y / h, "z": lm.z,
                        "visibility": lm.visibility if hasattr(lm, "visibility") else 1.0
                    })
                return adjusted_lms
    return None

def interpolate_gaps(frames, hand_key, source_key, max_interp_gap):
    valid_indices = []
    valid_pts = []
    for i, f in enumerate(frames):
        if f[source_key] in ["detected", "roi_fallback", "kalman_predicted"] and len(f[hand_key]) == 21:
            valid_indices.append(i)
            valid_pts.append([[p['x'], p['y'], p['z']] for p in f[hand_key]])
            
    if not valid_indices: return frames
    
    valid_indices = np.array(valid_indices)
    valid_pts = np.array(valid_pts)
    
    for i in range(21):
        for j in range(3):
            interp_func = interp1d(valid_indices, valid_pts[:, i, j], kind='linear', fill_value="extrapolate")
            for idx in range(len(frames)):
                if frames[idx][source_key] == "missing":
                    left_val = valid_indices[valid_indices < idx]
                    right_val = valid_indices[valid_indices > idx]
                    if len(left_val) > 0 and len(right_val) > 0:
                        dist = right_val[0] - left_val[-1]
                        if dist <= max_interp_gap:
                            if len(frames[idx][hand_key]) == 0:
                                frames[idx][hand_key] = [{"x":0, "y":0, "z":0, "visibility":0.5} for _ in range(21)]
                            frames[idx][hand_key][i][list('xyz')[j]] = float(interp_func(idx))
                            frames[idx][source_key] = "interpolated"
    return frames

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def get_finger_color(idx):
    if idx in [1, 2, 3, 4]: return (0, 140, 255)
    if idx in [5, 6, 7, 8]: return (0, 255, 0)
    if idx in [9, 10, 11, 12]: return (255, 0, 0)
    if idx in [13, 14, 15, 16]: return (0, 255, 255)
    if idx in [17, 18, 19, 20]: return (255, 0, 255)
    return (255, 255, 255)

def draw_custom_mesh(img, lms, source_color, width, height):
    if not lms or len(lms) < 21: return
    pts = []
    for lm in lms:
        pts.append((int(lm['x'] * width), int(lm['y'] * height)))
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(img, pts[start_idx], pts[end_idx], source_color, 2)
    for i, pt in enumerate(pts):
        cv2.circle(img, pt, 4, get_finger_color(i), -1)
        cv2.circle(img, pt, 4, (0, 0, 0), 1)

def process_video(input_path, output_dir, yolo_model, mp_hands, hands_fallback, config, device_type):
    filename = os.path.basename(input_path)
    base_name = os.path.splitext(filename)[0]
    output_video_path = os.path.join(output_dir, f"annotated_{filename}")
    output_json_path = os.path.join(output_dir, f"keypoints_{base_name}.json")

    print(f"Processing {filename}...")
    
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_container = av.open(output_video_path, mode='w', format='mp4', options={'movflags': 'faststart'})
    output_stream = output_container.add_stream('libx264', rate=int(fps))
    output_stream.width = width
    output_stream.height = height
    output_stream.pix_fmt = 'yuv420p'

    keypoint_data = {
        "metadata": {"filename": filename, "fps": fps, "total_frames": total_frames, "coverage": {"Left": {}, "Right": {}}},
        "frames": [],
    }

    kalman_l = KalmanHandTracker()
    kalman_r = KalmanHandTracker()
    raw_frames = []
    frame_idx = 0

    hands_to_track = config.get("hands_to_track", "Both")
    max_kalman_frames = config.get("max_kalman_frames", 15)
    max_interp_gap = config.get("max_interp_gap", 30)

    start_time = time.time()

    # Pass 1: Extraction
    while cap.isOpened():
        success, image = cap.read()
        if not success: break
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        frame_data = {
            "frame_index": frame_idx, "timestamp_sec": frame_idx / fps,
            "left_hand": [], "right_hand": [], "source_l": "missing", "source_r": "missing"
        }

        # Stage 1: MediaPipe Hands
        # Since mp_hands is the actual Hands instance passed in, we just use it
        results_h = mp_hands.process(image_rgb)
        found_l, found_r = False, False
        
        if results_h.multi_hand_landmarks and results_h.multi_handedness:
            for hlms, hness in zip(results_h.multi_hand_landmarks, results_h.multi_handedness):
                label = hness.classification[0].label
                if hness.classification[0].score < 0.6: continue
                    
                lms = serialize_landmarks(hlms)
                if label == 'Left' and hands_to_track in ["Both", "Left Only"] and not found_l:
                    frame_data["left_hand"] = lms
                    frame_data["source_l"] = "detected"
                    kalman_l.update(lms)
                    found_l = True
                elif label == 'Right' and hands_to_track in ["Both", "Right Only"] and not found_r:
                    frame_data["right_hand"] = lms
                    frame_data["source_r"] = "detected"
                    kalman_r.update(lms)
                    found_r = True

        # Stage 2: YOLO Pose ROI Fallback
        if yolo_model is not None and (not found_l or not found_r):
            yolo_results = yolo_model(image_rgb, device=device_type, verbose=False)
            if len(yolo_results) > 0 and yolo_results[0].keypoints is not None:
                kpts_norm = yolo_results[0].keypoints.xyn.cpu().numpy()
                kpts_conf = yolo_results[0].keypoints.conf.cpu().numpy() if yolo_results[0].keypoints.conf is not None else None
                
                for p_idx in range(len(kpts_norm)):
                    p_kpts = kpts_norm[p_idx]
                    p_conf = kpts_conf[p_idx] if kpts_conf is not None else np.ones(17)
                    
                    if hands_to_track in ["Both", "Left Only"] and not found_l and p_conf[9] > 0.6:
                        wx, wy = float(p_kpts[9][0]), float(p_kpts[9][1])
                        if wx > 0 and wy > 0:
                            ex = float(p_kpts[7][0]) if p_conf[7] > 0.5 else None
                            ey = float(p_kpts[7][1]) if p_conf[7] > 0.5 else None
                            
                            lms = crop_and_detect_hand(image_rgb, wx, wy, hands_fallback, target_label='Left', elbow_x=ex, elbow_y=ey)
                            if lms:
                                frame_data["left_hand"] = lms
                                frame_data["source_l"] = "roi_fallback"
                                kalman_l.update(lms)
                                found_l = True
                                break
                                
                for p_idx in range(len(kpts_norm)):
                    p_kpts = kpts_norm[p_idx]
                    p_conf = kpts_conf[p_idx] if kpts_conf is not None else np.ones(17)
                    
                    if hands_to_track in ["Both", "Right Only"] and not found_r and p_conf[10] > 0.6:
                        wx, wy = float(p_kpts[10][0]), float(p_kpts[10][1])
                        if wx > 0 and wy > 0:
                            ex = float(p_kpts[8][0]) if p_conf[8] > 0.5 else None
                            ey = float(p_kpts[8][1]) if p_conf[8] > 0.5 else None
                            
                            lms = crop_and_detect_hand(image_rgb, wx, wy, hands_fallback, target_label='Right', elbow_x=ex, elbow_y=ey)
                            if lms:
                                frame_data["right_hand"] = lms
                                frame_data["source_r"] = "roi_fallback"
                                kalman_r.update(lms)
                                found_r = True
                                break

        # Stage 3: Kalman
        if not found_l and kalman_l.is_initialized and kalman_l.frames_since_update < max_kalman_frames:
            pred = kalman_l.predict()
            if pred:
                frame_data["left_hand"] = pred
                frame_data["source_l"] = "kalman_predicted"
                
        if not found_r and kalman_r.is_initialized and kalman_r.frames_since_update < max_kalman_frames:
            pred = kalman_r.predict()
            if pred:
                frame_data["right_hand"] = pred
                frame_data["source_r"] = "kalman_predicted"

        raw_frames.append(frame_data)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"  Extracted {frame_idx}/{total_frames} frames...")

    cap.release()

    # Pass 2: Interpolation
    print(f"  Interpolating missing data (max gap {max_interp_gap} frames)...")
    if max_interp_gap > 0:
        raw_frames = interpolate_gaps(raw_frames, "left_hand", "source_l", max_interp_gap)
        raw_frames = interpolate_gaps(raw_frames, "right_hand", "source_r", max_interp_gap)

    # Pass 3: Rendering
    print(f"  Rendering annotated video...")
    cap = cv2.VideoCapture(input_path)
    frame_idx = 0
    color_map = {
        "detected": (0, 255, 0), "roi_fallback": (255, 255, 0), 
        "kalman_predicted": (0, 165, 255), "interpolated": (255, 0, 255)
    }

    while cap.isOpened():
        success, image = cap.read()
        if not success: break
        
        if frame_idx < len(raw_frames):
            f = raw_frames[frame_idx]
            if f["source_l"] != "missing":
                draw_custom_mesh(image, f["left_hand"], color_map.get(f["source_l"], (255,255,255)), width, height)
            if f["source_r"] != "missing":
                draw_custom_mesh(image, f["right_hand"], color_map.get(f["source_r"], (255,255,255)), width, height)
                
            cv2.putText(image, f"L: {f['source_l']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(f["source_l"], (0,0,255)), 2)
            cv2.putText(image, f"R: {f['source_r']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(f["source_r"], (0,0,255)), 2)
            
        frame_out = av.VideoFrame.from_ndarray(image, format='bgr24')
        for packet in output_stream.encode(frame_out):
            output_container.mux(packet)
            
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  Rendered {frame_idx}/{total_frames} frames...")

    cap.release()
    for packet in output_stream.encode():
        output_container.mux(packet)
    output_container.close()

    # Coverage Stats
    for side, key in [("Left", "source_l"), ("Right", "source_r")]:
        sources = [f[key] for f in raw_frames]
        total = len(sources)
        det = sources.count("detected") + sources.count("roi_fallback")
        recov = sources.count("kalman_predicted") + sources.count("interpolated")
        keypoint_data["metadata"]["coverage"][side] = {
            "detected_percent": det / total * 100 if total else 0,
            "recovered_percent": recov / total * 100 if total else 0,
            "missing_percent": sources.count("missing") / total * 100 if total else 0
        }

    keypoint_data["frames"] = raw_frames
    with open(output_json_path, "w") as f:
        json.dump(keypoint_data, f, indent=2)
        
    print(f"  Finished {filename} in {time.time()-start_time:.1f}s")
    print(f"    Left Det: {keypoint_data['metadata']['coverage']['Left']['detected_percent']:.1f}% | Rec: {keypoint_data['metadata']['coverage']['Left']['recovered_percent']:.1f}%")
    print(f"    Right Det: {keypoint_data['metadata']['coverage']['Right']['detected_percent']:.1f}% | Rec: {keypoint_data['metadata']['coverage']['Right']['recovered_percent']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Batch process hand keypoints using YOLO + MediaPipe.")
    parser.add_argument("input_dir", type=str, help="Directory containing input videos")
    parser.add_argument("output_dir", type=str, help="Directory to save output files")
    parser.add_argument("--hands", type=str, choices=["Both", "Left Only", "Right Only"], default="Both", help="Restrict tracking")
    parser.add_argument("--kalman", type=int, default=15, help="Max Kalman frames")
    parser.add_argument("--interp", type=int, default=30, help="Max Interpolation gap frames")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Initializing YOLO and MediaPipe models...")
    device_type = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
    print(f"Hardware Accelerator Selected: {device_type.upper()}")
    
    yolo_model = None
    try:
        yolo_model = YOLO("yolo26n-pose.pt")
    except Exception as e:
        print(f"Failed to load YOLO model: {e}")
        print("Running with standard MediaPipe only.")

    mp_hands_module = mp.solutions.hands
    hands_tracker = mp_hands_module.Hands(
        static_image_mode=False, max_num_hands=2,
        model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    
    hands_fallback = mp_hands_module.Hands(
        static_image_mode=False, max_num_hands=2,
        model_complexity=1, min_detection_confidence=0.15, min_tracking_confidence=0.15
    )

    valid_exts = [".mp4", ".mov", ".avi", ".webm"]
    files = [f for f in os.listdir(args.input_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    
    print(f"Found {len(files)} videos to process.")
    
    config = {
        "hands_to_track": args.hands,
        "max_kalman_frames": args.kalman,
        "max_interp_gap": args.interp
    }

    for f in files:
        input_path = os.path.join(args.input_dir, f)
        process_video(input_path, args.output_dir, yolo_model, hands_tracker, hands_fallback, config, device_type)

    hands_tracker.close()
    hands_fallback.close()
    print("Batch processing complete!")

if __name__ == "__main__":
    main()
