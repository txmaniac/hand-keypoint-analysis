import av
import cv2
import streamlit as st
import mediapipe as mp
import tempfile
import os
import json
import time
import threading
import pandas as pd
import plotly.express as px
import numpy as np
import torch
import zipfile
import io
from scipy.interpolate import interp1d
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

# -----------------------------
# Initialize MediaPipe
# -----------------------------
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Hand connections for drawing mesh
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def get_finger_color(idx):
    if idx in [1, 2, 3, 4]: return (0, 140, 255) # Thumb (Orange BGR)
    if idx in [5, 6, 7, 8]: return (0, 255, 0) # Index (Green)
    if idx in [9, 10, 11, 12]: return (255, 0, 0) # Middle (Blue)
    if idx in [13, 14, 15, 16]: return (0, 255, 255) # Ring (Yellow)
    if idx in [17, 18, 19, 20]: return (255, 0, 255) # Pinky (Magenta)
    return (255, 255, 255) # Palm (White)

def draw_custom_mesh(img, lms, source_color, width, height):
    if not lms or len(lms) < 21: return
    
    pts = []
    for lm in lms:
        pts.append((int(lm['x'] * width), int(lm['y'] * height)))
        
    # Draw lines colored by source
    for start_idx, end_idx in HAND_CONNECTIONS:
        cv2.line(img, pts[start_idx], pts[end_idx], source_color, 2)
        
    # Draw points colored by finger
    for i, pt in enumerate(pts):
        cv2.circle(img, pt, 4, get_finger_color(i), -1)
        cv2.circle(img, pt, 4, (0, 0, 0), 1) # Outline for contrast

# Initialize YOLO
from ultralytics import YOLO

# Determine optimal device for YOLO
device_type = 'cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')

st.set_page_config(page_title="Holistic Analyics Engine", page_icon="📈", layout="wide")

st.title("Holistic Keypoint Analytics System (Pipeline V2)")
st.markdown(
    "Analyze full body dynamics, hand interactions, and compare trajectories across multiple elicitation video studies with multi-stage dropout recovery."
)

# -----------------------------
# Sidebar configuration
# -----------------------------
st.sidebar.header("Configuration")

hands_to_track = st.sidebar.radio("Hands to Track", ["Both", "Left Only", "Right Only"], help="Restrict tracking to a specific hand to prevent hallucinating the other hand from background noise.")

# Toggle to bypass holistic model processing entirely
enable_body_pose = st.sidebar.toggle(
    "Enable Body Pose", 
    value=True, 
    help="When enabled, runs Pose model and uses wrists for ROI hand fallback."
)

flip_handedness = st.sidebar.toggle(
    "POV Toggle", 
    value=False, 
    help="Enable this if Left and Right hands are being swapped. Common with selfie-camera videos or mirrored recordings."
)

model_complexity = st.sidebar.selectbox("Model Complexity", [0, 1, 2], index=1, help="0 is fastest, 2 is most accurate but slowest (1 is baseline).")
min_detection_confidence = st.sidebar.slider("Min Detection Confidence", 0.0, 1.0, 0.5, help="Increase if you see random background objects detected as hands. Decrease if hands are genuinely missed.")
min_tracking_confidence = st.sidebar.slider("Min Tracking Confidence", 0.0, 1.0, 0.5)

st.sidebar.header("Recovery Pipeline")
max_kalman_frames = st.sidebar.slider("Max Kalman Extrapolation", 0, 30, 15, help="Number of frames to predict using constant velocity when detection fails.")
max_interp_gap = st.sidebar.slider("Max Interpolation Gap", 0, 60, 30, help="Maximum gap length (in frames) to bridge with linear interpolation post-processing.")


tab1, tab2, tab3, tab4 = st.tabs(["Upload Video", "Live Webcam", "Comparative Analytics Dashboard", "JSON to Video Converter"])

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

thread_local = threading.local()

def serialize_landmarks(landmark_list):
    """Safely converts mediapipe landmarks to a standard dictionary list."""
    if not landmark_list:
        return []
    return [
        {
            "x": float(lm.x),
            "y": float(lm.y),
            "z": float(lm.z),
            "visibility": float(lm.visibility) if hasattr(lm, "visibility") else 1.0,
        }
        for lm in landmark_list.landmark
    ]

class KalmanHandTracker:
    def __init__(self, num_points=21):
        self.num_points = num_points
        self.state = np.zeros((num_points, 6)) # x, y, z, vx, vy, vz
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
            # Apply velocity dampening (friction) so points don't fly off screen when tracking is lost
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
    """Fallback: crops around wrist and runs hands model."""
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

def _analyze_frame(image_bgr):
    """Used by webcam stream only (simplified)."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    if not hasattr(thread_local, "hands"):
        thread_local.hands = mp_hands.Hands(
            static_image_mode=False, max_num_hands=2,
            model_complexity=model_complexity if model_complexity < 2 else 1,
            min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence
        )
        
    results = thread_local.hands.process(img_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    return {}

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    _analyze_frame(img)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------
# Live webcam tab
# -----------------------------
with tab2:
    st.markdown("### Webcam Stream (Dynamic Engine)")
    webrtc_streamer(
        key="dynamic-tracking",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# -----------------------------
# Upload video tab
# -----------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "webm"])

    if uploaded_file is not None:
        if 'preview_video_name' not in st.session_state or st.session_state['preview_video_name'] != uploaded_file.name:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()
            st.session_state['preview_video_path'] = tfile.name
            st.session_state['preview_video_name'] = uploaded_file.name
            uploaded_file.seek(0)
            
        input_video_path = st.session_state['preview_video_path']
        
        cap_preview = cv2.VideoCapture(input_video_path)
        success, preview_frame = cap_preview.read()
        cap_preview.release()

        if success:
            preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            h, w, _ = preview_rgb.shape

            crop_margins = st.slider("Crop X-Axis Margins (%)", 0.0, 100.0, (0.0, 100.0))

            left_px = int(w * (crop_margins[0] / 100.0))
            right_px = int(w * (crop_margins[1] / 100.0))

            if left_px < right_px:
                preview_display = preview_rgb.copy()
                if left_px > 0: preview_display[:, :left_px] = preview_display[:, :left_px] // 3
                if right_px < w: preview_display[:, right_px:] = preview_display[:, right_px:] // 3
                thickness = max(2, w // 250)
                cv2.line(preview_display, (left_px, 0), (left_px, h), (255, 0, 0), thickness)
                cv2.line(preview_display, (right_px, 0), (right_px, h), (255, 0, 0), thickness)
                st.image(preview_display, caption="Live Crop Preview", use_container_width=True)
            else:
                st.error("Invalid crop margins.")

        if st.button("Start File Analysis"):
            with st.spinner("Executing Pipeline V2..."):
                output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                output_json_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name

                cap = cv2.VideoCapture(input_video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                crop_left_px = int(width * (crop_margins[0] / 100.0))
                crop_right_px = int(width * (crop_margins[1] / 100.0))
                if crop_left_px >= crop_right_px:
                    crop_left_px, crop_right_px = 0, width
                
                new_width = crop_right_px - crop_left_px
                if new_width % 2 != 0:
                    new_width -= 1
                    crop_right_px -= 1

                output_container = av.open(output_video_path, mode='w', format='mp4', options={'movflags': 'faststart'})
                output_stream = output_container.add_stream('libx264', rate=int(fps))
                output_stream.width = new_width
                output_stream.height = height
                output_stream.pix_fmt = 'yuv420p'

                keypoint_data = {
                    "metadata": {
                        "filename": uploaded_file.name,
                        "fps": fps,
                        "total_frames": total_frames,
                        "width": new_width,
                        "height": height,
                        "coverage": {"Left": {}, "Right": {}}
                    },
                    "frames": [],
                }

                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()
                
                hands = mp_hands.Hands(
                    static_image_mode=False, max_num_hands=2,
                    model_complexity=model_complexity if model_complexity < 2 else 1,
                    min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence
                )
                
                hands_fallback = mp_hands.Hands(
                    static_image_mode=False, max_num_hands=2,
                    model_complexity=model_complexity if model_complexity < 2 else 1,
                    min_detection_confidence=0.15, min_tracking_confidence=0.15
                )
                
                yolo_model = None
                if enable_body_pose:
                    try:
                        yolo_model = YOLO("yolo26n-pose.pt")
                    except Exception as e:
                        st.error(f"Failed to load YOLO model: {e}")
                        enable_body_pose = False

                kalman_l = KalmanHandTracker()
                kalman_r = KalmanHandTracker()

                raw_frames = []

                frame_idx = 0
                while cap.isOpened():
                    success, image = cap.read()
                    if not success: break

                    image = image[:, crop_left_px:crop_right_px]
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    frame_data = {
                        "frame_index": frame_idx,
                        "timestamp_sec": frame_idx / fps,
                        "pose": [],
                        "left_hand": [],
                        "right_hand": [],
                        "source_l": "missing",
                        "source_r": "missing"
                    }

                    # Stage 1: Primary Hands Model
                    results_h = hands.process(image_rgb)
                    found_l, found_r = False, False
                    
                    if results_h.multi_hand_landmarks and results_h.multi_handedness:
                        for hlms, hness in zip(results_h.multi_hand_landmarks, results_h.multi_handedness):
                            label = hness.classification[0].label
                            if flip_handedness:
                                label = 'Right' if label == 'Left' else 'Left'
                                
                            score = hness.classification[0].score
                            
                            # Filter weak handedness to avoid swapping or background noise
                            if score < 0.6: 
                                continue
                                
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

                    # Stage 2: YOLO Pose & ROI Fallback
                    if enable_body_pose and yolo_model is not None:
                        # YOLO runs on the un-cropped image to get better global context, or cropped?
                        # Since all our coordinates are relative to the cropped image, we should run YOLO on the cropped image
                        # just like MediaPipe. This avoids complex coordinate translations.
                        yolo_results = yolo_model(image_rgb, device=device_type, verbose=False)
                        
                        if len(yolo_results) > 0 and yolo_results[0].keypoints is not None:
                            # xyn is normalized coordinates [0, 1]
                            kpts_norm = yolo_results[0].keypoints.xyn.cpu().numpy()
                            kpts_conf = yolo_results[0].keypoints.conf.cpu().numpy() if yolo_results[0].keypoints.conf is not None else None
                            
                            # Can have multiple people
                            for p_idx in range(len(kpts_norm)):
                                p_kpts = kpts_norm[p_idx]
                                p_conf = kpts_conf[p_idx] if kpts_conf is not None else np.ones(17)
                                
                                # Left Wrist = 9, Right Wrist = 10
                                # If flipped, we swap the YOLO indices
                                left_wrist_idx = 10 if flip_handedness else 9
                                right_wrist_idx = 9 if flip_handedness else 10
                                
                                left_elbow_idx = 8 if flip_handedness else 7
                                right_elbow_idx = 7 if flip_handedness else 8
                                
                                if hands_to_track in ["Both", "Left Only"] and not found_l and p_conf[left_wrist_idx] > 0.6:
                                    wx, wy = float(p_kpts[left_wrist_idx][0]), float(p_kpts[left_wrist_idx][1])
                                    if wx > 0 and wy > 0:
                                        ex = float(p_kpts[left_elbow_idx][0]) if p_conf[left_elbow_idx] > 0.5 else None
                                        ey = float(p_kpts[left_elbow_idx][1]) if p_conf[left_elbow_idx] > 0.5 else None
                                        
                                        target = 'Right' if flip_handedness else 'Left'
                                        lms = crop_and_detect_hand(image_rgb, wx, wy, hands_fallback, target_label=target, elbow_x=ex, elbow_y=ey)
                                        if lms:
                                            frame_data["left_hand"] = lms
                                            frame_data["source_l"] = "roi_fallback"
                                            kalman_l.update(lms)
                                            found_l = True
                                            break # found left, stop checking other people
                                            
                            for p_idx in range(len(kpts_norm)):
                                p_kpts = kpts_norm[p_idx]
                                p_conf = kpts_conf[p_idx] if kpts_conf is not None else np.ones(17)
                                
                                if hands_to_track in ["Both", "Right Only"] and not found_r and p_conf[right_wrist_idx] > 0.6:
                                    wx, wy = float(p_kpts[right_wrist_idx][0]), float(p_kpts[right_wrist_idx][1])
                                    if wx > 0 and wy > 0:
                                        ex = float(p_kpts[right_elbow_idx][0]) if p_conf[right_elbow_idx] > 0.5 else None
                                        ey = float(p_kpts[right_elbow_idx][1]) if p_conf[right_elbow_idx] > 0.5 else None
                                        
                                        target = 'Left' if flip_handedness else 'Right'
                                        lms = crop_and_detect_hand(image_rgb, wx, wy, hands_fallback, target_label=target, elbow_x=ex, elbow_y=ey)
                                        if lms:
                                            frame_data["right_hand"] = lms
                                            frame_data["source_r"] = "roi_fallback"
                                            kalman_r.update(lms)
                                            found_r = True
                                            break # found right, stop checking other people

                    # Stage 3: Kalman Prediction
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

                    if found_l and frame_data["source_l"] in ["detected", "roi_fallback"]:
                        # Convert back to hand landmarks for drawing
                        from google.protobuf.json_format import ParseDict
                        from mediapipe.framework.formats import landmark_pb2
                        # Draw is deferred to after interpolation if we want to draw everything, but for video output we draw what we have now.
                        # Actually, we will save the raw_frames, run interpolation, and THEN render video.
                        pass 

                    raw_frames.append(frame_data)

                    frame_idx += 1
                    if frame_idx % max(1, (total_frames // 100)) == 0:
                        progress_bar.progress(min(frame_idx / total_frames * 0.4, 0.4))
                        status_text.text(f"Pass 1 (Detection): {frame_idx}/{total_frames} frames")

                cap.release()
                hands.close()
                hands_fallback.close()
                
                # Stage 4: Interpolation Post-Processing
                status_text.text("Pass 2 (Interpolation)...")
                
                def interpolate_gaps(frames, hand_key, source_key):
                    valid_indices = []
                    valid_pts = []
                    for i, f in enumerate(frames):
                        if f[source_key] in ["detected", "roi_fallback", "kalman_predicted"] and len(f[hand_key]) == 21:
                            valid_indices.append(i)
                            valid_pts.append([[p['x'], p['y'], p['z']] for p in f[hand_key]])
                            
                    if not valid_indices: return frames
                    
                    valid_indices = np.array(valid_indices)
                    valid_pts = np.array(valid_pts) # shape (N, 21, 3)
                    
                    for i in range(21):
                        for j in range(3):
                            # interp1d for each coordinate of each landmark
                            interp_func = interp1d(valid_indices, valid_pts[:, i, j], kind='linear', fill_value="extrapolate")
                            
                            # Find gaps
                            for idx in range(len(frames)):
                                if frames[idx][source_key] == "missing":
                                    # Find nearest valid indices
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

                if max_interp_gap > 0:
                    raw_frames = interpolate_gaps(raw_frames, "left_hand", "source_l")
                    raw_frames = interpolate_gaps(raw_frames, "right_hand", "source_r")

                progress_bar.progress(0.6)
                
                # Render Output Video
                status_text.text("Pass 3 (Rendering Video)...")
                cap = cv2.VideoCapture(input_video_path)
                frame_idx = 0
                
                # Using the global draw_custom_mesh function

                while cap.isOpened():
                    success, image = cap.read()
                    if not success: break
                    image = image[:, crop_left_px:crop_right_px]
                    
                    if frame_idx < len(raw_frames):
                        f = raw_frames[frame_idx]
                        
                        color_map = {
                            "detected": (0, 255, 0), # Green
                            "roi_fallback": (255, 255, 0), # Cyan
                            "kalman_predicted": (0, 165, 255), # Orange
                            "interpolated": (255, 0, 255) # Purple
                        }
                        
                        if f["source_l"] != "missing":
                            draw_custom_mesh(image, f["left_hand"], color_map.get(f["source_l"], (255,255,255)), new_width, height)
                        if f["source_r"] != "missing":
                            draw_custom_mesh(image, f["right_hand"], color_map.get(f["source_r"], (255,255,255)), new_width, height)
                            
                        # Optional: Add text overlay
                        cv2.putText(image, f"L: {f['source_l']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(f["source_l"], (0,0,255)), 2)
                        cv2.putText(image, f"R: {f['source_r']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(f["source_r"], (0,0,255)), 2)
                        
                    frame_out = av.VideoFrame.from_ndarray(image, format='bgr24')
                    for packet in output_stream.encode(frame_out):
                        output_container.mux(packet)
                        
                    frame_idx += 1
                    if frame_idx % max(1, (total_frames // 100)) == 0:
                        progress_bar.progress(0.6 + min(frame_idx / total_frames * 0.4, 0.4))

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

                st.session_state['processed_video_path'] = output_video_path
                st.session_state['processed_json_path'] = output_json_path
                st.session_state['processed_file_name'] = uploaded_file.name
                st.session_state['coverage_data'] = keypoint_data["metadata"]["coverage"]
                
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")

        if st.session_state.get('processed_file_name') == uploaded_file.name:
            out_vid = st.session_state['processed_video_path']
            out_json = st.session_state['processed_json_path']
            cov = st.session_state.get('coverage_data', {})
            
            # Show coverage
            st.markdown("### Detection Coverage")
            col1, col2 = st.columns(2)
            for side, data in cov.items():
                col = col1 if side == "Left" else col2
                with col:
                    st.markdown(f"**{side} Hand**")
                    st.progress(data["detected_percent"] / 100, text=f"Detected: {data['detected_percent']:.1f}%")
                    st.progress(data["recovered_percent"] / 100, text=f"Recovered: {data['recovered_percent']:.1f}%")
                    st.progress(data["missing_percent"] / 100, text=f"Missing: {data['missing_percent']:.1f}%")

            st.video(out_vid)
            col1, col2 = st.columns(2)
            base_fname = os.path.splitext(uploaded_file.name)[0]
            
            with col1:
                with open(out_vid, "rb") as f:
                    st.download_button("Download Annotated Video (MP4)", data=f, file_name=f"processed_{base_fname}.mp4", mime="video/mp4")
            with col2:
                with open(out_json, "r") as f:
                    st.download_button("Download Keypoint Data (JSON)", data=f, file_name=f"processed_{base_fname}.json", mime="application/json")


# -----------------------------
# Analytics Dashboard Tab
# -----------------------------
with tab3:
    st.markdown("### 📊 Comparative Analysis Dashboard")
    st.info("Upload multiple generated `.json` keypoint files to compare biomechanical tracks side-by-side.")
    
    comparative_files = st.file_uploader("Upload Output JSON datasets", type=["json"], accept_multiple_files=True)
    
    if comparative_files:
        st.divider()
        dfs_all = []
        coverage_rows = []
        
        for file in comparative_files:
            data = json.load(file)
            fname = data["metadata"]["filename"]
            fps = data["metadata"]["fps"]
            cov = data["metadata"].get("coverage", {})
            
            if cov:
                coverage_rows.append({
                    "Video": fname,
                    "Left Hand: Directly Detected (%)": cov.get("Left", {}).get("detected_percent", 0),
                    "Left Hand: Mathematically Recovered (%)": cov.get("Left", {}).get("recovered_percent", 0),
                    "Right Hand: Directly Detected (%)": cov.get("Right", {}).get("detected_percent", 0),
                    "Right Hand: Mathematically Recovered (%)": cov.get("Right", {}).get("recovered_percent", 0),
                })
            
            records = []
            
            for frame in data["frames"]:
                t = frame["timestamp_sec"]
                pose = frame.get("pose", [])
                lh = frame.get("left_hand", [])
                rh = frame.get("right_hand", [])
                
                l_vis = 1 if frame.get("source_l", "missing") != "missing" else 0
                r_vis = 1 if frame.get("source_r", "missing") != "missing" else 0
                
                def dist3d(a, b):
                    return np.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2 + (a['z']-b['z'])**2)
                
                inter_hand_dist = dist3d(lh[0], rh[0]) if l_vis and r_vis and len(lh)>0 and len(rh)>0 else np.nan
                
                l_pinch = dist3d(lh[4], lh[8]) if l_vis and len(lh) > 8 else np.nan
                r_pinch = dist3d(rh[4], rh[8]) if r_vis and len(rh) > 8 else np.nan
                
                def get_roll(hand):
                    if len(hand) > 17:
                        dx = hand[17]['x'] - hand[5]['x']
                        dy = hand[17]['y'] - hand[5]['y']
                        return np.degrees(np.arctan2(dy, dx))
                    return np.nan
                    
                l_roll = get_roll(lh) if l_vis else np.nan
                r_roll = get_roll(rh) if r_vis else np.nan
                
                records.append({
                    "Time (s)": t,
                    "Video": fname,
                    "Left Hand Presence": l_vis,
                    "Right Hand Presence": r_vis,
                    "Inter-Hand Distance": inter_hand_dist,
                    "Left Pinch": l_pinch,
                    "Right Pinch": r_pinch,
                    "L_Wx": lh[0]['x'] if l_vis and len(lh)>0 else np.nan,
                    "L_Wy": lh[0]['y'] if l_vis and len(lh)>0 else np.nan,
                    "L_Wz": lh[0]['z'] if l_vis and len(lh)>0 else np.nan,
                    "R_Wx": rh[0]['x'] if r_vis and len(rh)>0 else np.nan,
                    "R_Wy": rh[0]['y'] if r_vis and len(rh)>0 else np.nan,
                    "R_Wz": rh[0]['z'] if r_vis and len(rh)>0 else np.nan,
                    "L_Roll": l_roll,
                    "R_Roll": r_roll,
                    "Source_L": frame.get("source_l", "missing"),
                    "Source_R": frame.get("source_r", "missing")
                })
            
            df = pd.DataFrame(records)
            
            # Velocity Calculation
            df['dt'] = df['Time (s)'].diff().fillna(1.0/fps)
            df['L_Vel'] = np.sqrt(df['L_Wx'].diff()**2 + df['L_Wy'].diff()**2 + df['L_Wz'].diff()**2) / df['dt']
            df['R_Vel'] = np.sqrt(df['R_Wx'].diff()**2 + df['R_Wy'].diff()**2 + df['R_Wz'].diff()**2) / df['dt']
            df['Inter_Vel'] = df['Inter-Hand Distance'].diff() / df['dt']
            
            # Acceleration
            df['L_Acc'] = df['L_Vel'].diff() / df['dt']
            df['R_Acc'] = df['R_Vel'].diff() / df['dt']
            
            # Jerk
            df['L_Jerk'] = df['L_Acc'].diff() / df['dt']
            df['R_Jerk'] = df['R_Acc'].diff() / df['dt']
            
            # Tremor (Smoothness) Calculation
            window_size = max(3, int(0.5 * fps))
            df['L_Tremor'] = df['L_Vel'].rolling(window=window_size, min_periods=1).std()
            df['R_Tremor'] = df['R_Vel'].rolling(window=window_size, min_periods=1).std()
            
            dfs_all.append(df)
            
        if coverage_rows:
            st.markdown("#### 0. Data Quality / Coverage Summary")
            cov_df = pd.DataFrame(coverage_rows)
            st.dataframe(cov_df, use_container_width=True)
            
        if dfs_all:
            final_df = pd.concat(dfs_all)
            
            import plotly.graph_objects as go
            import plotly.colors as pc
            
            def plot_tensorboard_style(df, x_col, y_col, smoothing_weight, title):
                fig = go.Figure()
                videos = df["Video"].unique()
                colors = pc.qualitative.Plotly
                
                for i, vid in enumerate(videos):
                    vid_df = df[df["Video"] == vid]
                    c = colors[i % len(colors)]
                    
                    # Raw Data (Faint)
                    fig.add_trace(go.Scatter(
                        x=vid_df[x_col], y=vid_df[y_col],
                        mode='lines', line=dict(color=c, width=1),
                        opacity=0.2, name=f"{vid} (Raw)",
                        showlegend=False
                    ))
                    
                    # Smoothed Data (Solid)
                    if smoothing_weight > 0:
                        smoothed = vid_df[y_col].ewm(alpha=1 - smoothing_weight, adjust=False).mean()
                    else:
                        smoothed = vid_df[y_col]
                        
                    fig.add_trace(go.Scatter(
                        x=vid_df[x_col], y=smoothed,
                        mode='lines', line=dict(color=c, width=2.5),
                        opacity=1.0, name=f"{vid}"
                    ))
                    
                fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col, hovermode="x unified")
                return fig
            
            st.markdown("#### 1. Bimanual Coordination")
            smooth_inter = st.slider("TensorBoard Smoothing (Inter-Hand)", 0.0, 0.99, 0.60, 0.05, key="smooth_inter")
            fig_inter = plot_tensorboard_style(final_df, "Time (s)", "Inter-Hand Distance", smooth_inter, "Distance Between Left and Right Hand")
            st.plotly_chart(fig_inter, use_container_width=True)
            
            # Add Source Strip
            st.markdown("#### Source Reliability Heatmap")
            
            plotly_color_map = {
                "detected": "#00FF00",          # Green
                "roi_fallback": "#FFFF00",      # Yellow
                "kalman_predicted": "#FFA500",  # Orange
                "interpolated": "#FF00FF",      # Magenta
                "missing": "#FF0000"            # Red
            }
            
            fig_source = px.scatter(final_df, x="Time (s)", y="Video", color="Source_L", title="Left Hand Detection Source", symbol="Source_L", color_discrete_map=plotly_color_map)
            fig_source.update_traces(marker=dict(size=5))
            st.plotly_chart(fig_source, use_container_width=True)
            
            fig_source_r = px.scatter(final_df, x="Time (s)", y="Video", color="Source_R", title="Right Hand Detection Source", symbol="Source_R", color_discrete_map=plotly_color_map)
            fig_source_r.update_traces(marker=dict(size=5))
            st.plotly_chart(fig_source_r, use_container_width=True)
            
            col_l, col_r = st.columns(2)
            
            with col_l:
                smooth_tl = st.slider("TensorBoard Smoothing (Left Jitter)", 0.0, 0.99, 0.60, 0.05, key="smooth_tl")
                fig_tremor_l = plot_tensorboard_style(final_df, "Time (s)", "L_Tremor", smooth_tl, "Left Movement Jitter")
                st.plotly_chart(fig_tremor_l, use_container_width=True)
                
                smooth_pl = st.slider("TensorBoard Smoothing (Left Pinch)", 0.0, 0.99, 0.60, 0.05, key="smooth_pl")
                fig_pinch_l = plot_tensorboard_style(final_df, "Time (s)", "Left Pinch", smooth_pl, "Left Pinch Grip Distance")
                st.plotly_chart(fig_pinch_l, use_container_width=True)

            with col_r:
                smooth_tr = st.slider("TensorBoard Smoothing (Right Jitter)", 0.0, 0.99, 0.60, 0.05, key="smooth_tr")
                fig_tremor_r = plot_tensorboard_style(final_df, "Time (s)", "R_Tremor", smooth_tr, "Right Movement Jitter")
                st.plotly_chart(fig_tremor_r, use_container_width=True)
                
                smooth_pr = st.slider("TensorBoard Smoothing (Right Pinch)", 0.0, 0.99, 0.60, 0.05, key="smooth_pr")
                fig_pinch_r = plot_tensorboard_style(final_df, "Time (s)", "Right Pinch", smooth_pr, "Right Pinch Grip Distance")
                st.plotly_chart(fig_pinch_r, use_container_width=True)

            st.markdown("#### 2. Advanced Biomechanics (Jerk & Orientation)")
            col_l2, col_r2 = st.columns(2)
            
            with col_l2:
                smooth_jerk_l = st.slider("TensorBoard Smoothing (Left Jerk)", 0.0, 0.99, 0.60, 0.05, key="smooth_jerk_l")
                fig_jerk_l = plot_tensorboard_style(final_df, "Time (s)", "L_Jerk", smooth_jerk_l, "Left Jerk (Motor Smoothness)")
                st.plotly_chart(fig_jerk_l, use_container_width=True)
                
                smooth_roll_l = st.slider("TensorBoard Smoothing (Left Roll)", 0.0, 0.99, 0.60, 0.05, key="smooth_roll_l")
                fig_roll_l = plot_tensorboard_style(final_df, "Time (s)", "L_Roll", smooth_roll_l, "Left Hand Pronation (Roll Angle)")
                st.plotly_chart(fig_roll_l, use_container_width=True)
                
            with col_r2:
                smooth_jerk_r = st.slider("TensorBoard Smoothing (Right Jerk)", 0.0, 0.99, 0.60, 0.05, key="smooth_jerk_r")
                fig_jerk_r = plot_tensorboard_style(final_df, "Time (s)", "R_Jerk", smooth_jerk_r, "Right Jerk (Motor Smoothness)")
                st.plotly_chart(fig_jerk_r, use_container_width=True)
                
                smooth_roll_r = st.slider("TensorBoard Smoothing (Right Roll)", 0.0, 0.99, 0.60, 0.05, key="smooth_roll_r")
                fig_roll_r = plot_tensorboard_style(final_df, "Time (s)", "R_Roll", smooth_roll_r, "Right Hand Pronation (Roll Angle)")
                st.plotly_chart(fig_roll_r, use_container_width=True)
                
            st.markdown("#### 3. Kinematic Phase Portraits")
            st.markdown("A Phase Portrait plots Distance against Velocity to reveal cyclical movement loops. Healthy deliberate movement forms clean circles, while erratic movement creates jagged scribbles.")
            fig_phase = px.line(final_df.dropna(subset=["Inter-Hand Distance", "Inter_Vel"]), 
                                x="Inter-Hand Distance", y="Inter_Vel", color="Video", 
                                title="Bimanual Phase Space (Distance vs Velocity)")
            st.plotly_chart(fig_phase, use_container_width=True)


# -----------------------------
# JSON to Video Tab
# -----------------------------
with tab4:
    st.markdown("### 🎞️ JSON to Video Converter (Batch Mode)")
    st.markdown("Upload multiple `keypoints.json` files to generate MP4 videos of the animated biomechanical model.")
    
    json_files = st.file_uploader("Upload keypoints JSON", type=["json"], accept_multiple_files=True)
    
    if json_files:
        st.markdown("#### Configuration")
        col1, col2 = st.columns(2)
        
        # Load the first file to get default metadata
        data_preview = json.load(json_files[0])
        json_files[0].seek(0) # reset pointer for later processing
        
        meta = data_preview.get("metadata", {})
        default_w = meta.get("width", 1920)
        default_h = meta.get("height", 1080)
        default_fps = meta.get("fps", 30.0)
        
        if "width" not in meta or "height" not in meta:
            st.warning("Original video resolution not found in the first JSON file. Using defaults. Please specify output resolution if known.")
            
        with col1:
            out_w = st.number_input("Output Width", min_value=128, max_value=3840, value=default_w)
            out_h = st.number_input("Output Height", min_value=128, max_value=2160, value=default_h)
            
        with col2:
            out_fps = st.number_input("Output FPS", min_value=1.0, max_value=120.0, value=float(default_fps))
            
        if st.button("Generate Videos"):
            with st.spinner(f"Generating {len(json_files)} video(s)..."):
                zip_buffer = io.BytesIO()
                
                overall_progress = st.progress(0)
                overall_status = st.empty()
                
                # Keep track of the last processed video path for preview
                last_video_path = None
                
                with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
                    for f_idx, j_file in enumerate(json_files):
                        overall_status.text(f"Processing file {f_idx+1}/{len(json_files)}: {j_file.name}")
                        data = json.load(j_file)
                        
                        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                        last_video_path = output_video_path
                        
                        output_container = av.open(output_video_path, mode='w', format='mp4', options={'movflags': 'faststart'})
                        output_stream = output_container.add_stream('libx264', rate=int(out_fps))
                        output_stream.width = out_w
                        output_stream.height = out_h
                        output_stream.pix_fmt = 'yuv420p'
                        
                        frames = data.get("frames", [])
                        
                        for f in frames:
                            # Create black background
                            image = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                            
                            color_map = {
                                "detected": (0, 255, 0), # Green
                                "roi_fallback": (255, 255, 0), # Cyan
                                "kalman_predicted": (0, 165, 255), # Orange
                                "interpolated": (255, 0, 255) # Purple
                            }
                            
                            source_l = f.get("source_l", "missing")
                            source_r = f.get("source_r", "missing")
                            
                            if source_l != "missing":
                                draw_custom_mesh(image, f.get("left_hand", []), color_map.get(source_l, (255,255,255)), out_w, out_h)
                            if source_r != "missing":
                                draw_custom_mesh(image, f.get("right_hand", []), color_map.get(source_r, (255,255,255)), out_w, out_h)
                                
                            cv2.putText(image, f"L: {source_l}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(source_l, (0,0,255)), 2)
                            cv2.putText(image, f"R: {source_r}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_map.get(source_r, (0,0,255)), 2)
                            
                            # Convert to AV Frame
                            frame_out = av.VideoFrame.from_ndarray(image, format='bgr24')
                            for packet in output_stream.encode(frame_out):
                                output_container.mux(packet)
                                
                        # Flush
                        for packet in output_stream.encode():
                            output_container.mux(packet)
                        output_container.close()
                        
                        # Add to ZIP
                        with open(output_video_path, "rb") as vf:
                            video_name = f"generated_{j_file.name.replace('.json', '.mp4')}"
                            zip_file.writestr(video_name, vf.read())
                            
                        overall_progress.progress((f_idx + 1) / len(json_files))
                        
                overall_status.text("Generation complete!")
                st.success(f"Generated {len(json_files)} video(s) Successfully!")
                
                if len(json_files) == 1 and last_video_path:
                    st.video(last_video_path)
                    with open(last_video_path, "rb") as vf:
                        st.download_button(
                            label="Download Generated Video (MP4)",
                            data=vf,
                            file_name=f"generated_{json_files[0].name.replace('.json', '.mp4')}",
                            mime="video/mp4"
                        )
                else:
                    if last_video_path:
                        st.markdown("### Preview (Last Processed File)")
                        st.video(last_video_path)
                    
                    zip_buffer.seek(0)
                    st.download_button(
                        label=f"Download All {len(json_files)} Videos (ZIP)",
                        data=zip_buffer,
                        file_name="generated_videos.zip",
                        mime="application/zip"
                    )


