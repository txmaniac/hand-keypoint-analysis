import av
import cv2
import streamlit as st
import mediapipe as mp
import tempfile
import os
import json
import time
import threading
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

# -----------------------------
# Optional YOLO Pose dependency
# -----------------------------
try:
    # Ultralytics YOLO (YOLOv8/YOLO11) pose models
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except Exception:
    YOLO = None
    _HAS_ULTRALYTICS = False

# -----------------------------
# Initialize MediaPipe Hands
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

st.set_page_config(page_title="Hand + Body Keypoint Analysis", page_icon="🖐️", layout="wide")

st.title("Hand Keypoint Analysis System")
st.markdown(
    "Analyze **hand keypoints** via uploaded video files or live webcam stream with 21-point finger tracking. "
    "Optionally enable **body pose** using your local YOLO pose model."
)

# -----------------------------
# Sidebar configuration
# -----------------------------
st.sidebar.header("Configuration")

# Streamlit compatibility: older versions may not have st.sidebar.toggle
_sidebar_toggle = getattr(st.sidebar, "toggle", st.sidebar.checkbox)

# Hand analysis is ALWAYS ON (default requirement).
# Body pose analysis is optional via this toggle.
enable_body_pose = _sidebar_toggle(
    "Enable Body Pose (YOLO)",
    value=False,
    help="When enabled, runs YOLO pose in addition to MediaPipe hands and records both in JSON."
)

# MediaPipe settings (hands)
model_complexity = st.sidebar.selectbox(
    "Hand Model Complexity (MediaPipe)",
    [0, 1],
    index=0,
    help="0 is smaller/faster. 1 is larger/slower."
)
min_detection_confidence = st.sidebar.slider("Hand Min Detection Confidence", 0.0, 1.0, 0.8)
min_tracking_confidence = st.sidebar.slider("Hand Min Tracking Confidence", 0.0, 1.0, 0.8)

# YOLO settings (pose)
if enable_body_pose:
    yolo_weights_path = st.sidebar.text_input(
        "YOLO Pose Weights Path",
        value="yolo26n-pose.pt",
        help="Path to your local YOLO pose weights (.pt). Example: yolov8n-pose.pt or custom best.pt"
    )
    yolo_conf = st.sidebar.slider("YOLO Pose Confidence", 0.0, 1.0, 0.25)
    yolo_iou = st.sidebar.slider("YOLO Pose IoU", 0.0, 1.0, 0.45)

    if not _HAS_ULTRALYTICS:
        st.sidebar.error(
            "Ultralytics is not installed. Install it to use YOLO pose:\n\n"
            "`pip install ultralytics`"
        )
else:
    # Placeholders so code can reference them safely.
    yolo_weights_path = ""
    yolo_conf = 0.25
    yolo_iou = 0.45

# -----------------------------
# App layout
# -----------------------------
tab1, tab2 = st.tabs(["Upload Video", "Live Webcam Analysis"])

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Thread locals: MediaPipe can freeze if created on a different thread than used.
thread_local = threading.local()

# YOLO lock: prevent concurrent inference calls from WebRTC threads.
yolo_lock = threading.Lock()


@st.cache_resource(show_spinner=False)
def _load_yolo_model(weights_path: str):
    """Load and cache the YOLO pose model (Ultralytics)."""
    if not _HAS_ULTRALYTICS:
        raise RuntimeError("Ultralytics is not available")
    if not weights_path:
        raise ValueError("YOLO weights path is empty")
    return YOLO(weights_path)


# COCO-17 skeleton edges for pose drawing (index pairs)
COCO17_SKELETON = [
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (0, 5), (0, 6),
]


def _run_mediapipe_hands_on_frame(image_bgr):
    """Annotate frame with MediaPipe hands and return structured landmarks."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Lazily initialize on the exact thread WebRTC uses
    if not hasattr(thread_local, "hands"):
        thread_local.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    results = thread_local.hands.process(img_rgb)

    hands_out = []
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(
                image_bgr,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            hand_info = {
                "label": handedness.classification[0].label,
                "score": float(handedness.classification[0].score),
                "landmarks": [
                    {
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "visibility": float(lm.visibility) if hasattr(lm, "visibility") else 1.0,
                    }
                    for lm in hand_landmarks.landmark
                ],
            }
            hands_out.append(hand_info)

    return hands_out


def _run_yolo_pose_on_frame(image_bgr, yolo_model):
    """Annotate frame with YOLO pose and return structured keypoints."""
    # Ultralytics accepts numpy arrays; keep BGR.
    with yolo_lock:
        pred = yolo_model.predict(image_bgr, conf=yolo_conf, iou=yolo_iou, verbose=False)

    if not pred:
        return []

    r = pred[0]
    persons_out = []

    if getattr(r, "keypoints", None) is None or r.keypoints is None:
        return persons_out

    xy = r.keypoints.xy
    conf = getattr(r.keypoints, "conf", None)

    # Move to CPU/numpy safely
    xy_np = xy.cpu().numpy() if hasattr(xy, "cpu") else xy
    conf_np = conf.cpu().numpy() if (conf is not None and hasattr(conf, "cpu")) else conf

    for i in range(xy_np.shape[0]):
        kpts = []
        for j in range(xy_np.shape[1]):
            x, y = float(xy_np[i, j, 0]), float(xy_np[i, j, 1])
            c = float(conf_np[i, j]) if conf_np is not None else 1.0
            kpts.append({"x": x, "y": y, "conf": c})

        persons_out.append({"person_index": i, "keypoints": kpts})

        # Draw keypoints
        for kp in kpts:
            if kp["conf"] >= yolo_conf:
                cv2.circle(image_bgr, (int(kp["x"]), int(kp["y"])), 3, (0, 255, 0), -1)

        # Draw skeleton
        for a, b in COCO17_SKELETON:
            if a < len(kpts) and b < len(kpts):
                if kpts[a]["conf"] >= yolo_conf and kpts[b]["conf"] >= yolo_conf:
                    pt1 = (int(kpts[a]["x"]), int(kpts[a]["y"]))
                    pt2 = (int(kpts[b]["x"]), int(kpts[b]["y"]))
                    cv2.line(image_bgr, pt1, pt2, (255, 0, 0), 2)

    return persons_out


def _analyze_frame(image_bgr):
    """Run hands always; optionally run body pose. Return structured data."""
    # Hands (always)
    hands = _run_mediapipe_hands_on_frame(image_bgr)

    pose_persons = None
    pose_error = None

    if enable_body_pose:
        if not _HAS_ULTRALYTICS:
            pose_error = "ultralytics_not_installed"
            pose_persons = []
        else:
            try:
                yolo_model = _load_yolo_model(yolo_weights_path)
                pose_persons = _run_yolo_pose_on_frame(image_bgr, yolo_model)
            except Exception as e:
                pose_error = str(e)
                pose_persons = []

    payload = {
        "hands": hands,
        "body_pose": {
            "enabled": bool(enable_body_pose),
            "persons": pose_persons if enable_body_pose else [],
            "error": pose_error,
            "weights": yolo_weights_path if enable_body_pose else None,
        },
    }
    return payload


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    _ = _analyze_frame(img)  # annotate in-place
    return av.VideoFrame.from_ndarray(img, format="bgr24")


# -----------------------------
# Live webcam tab
# -----------------------------
with tab2:
    st.markdown("### Webcam Stream")

    webrtc_streamer(
        key="hand-plus-pose",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,  # MediaPipe prefers sync; YOLO also OK.
    )

    st.caption(
        "Hands analysis is always on. Toggle **Enable Body Pose (YOLO)** in the sidebar to add body pose overlay."
    )


# -----------------------------
# Upload video tab
# -----------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "webm"])

    if uploaded_file is not None:
        st.markdown("### Uploaded Video")
        st.video(uploaded_file)

        if st.button("Start File Analysis"):
            with st.spinner("Processing video... This may take a while depending on the video length and resolution."):

                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                tfile.close()  # Release lock for Windows compatibility
                input_video_path = tfile.name

                output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                output_json_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name

                cap = cv2.VideoCapture(input_video_path)

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                if total_frames <= 0:
                    st.error("Could not read video frames. The file might be corrupted.")
                    cap.release()
                    st.stop()

                # Use avc1 codec (H.264) which natively plays in web browsers seamlessly
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                out = cv2.VideoWriter(output_video_path, fourcc, fps if fps > 0 else 25, (width, height))

                # If pose enabled, load once here for faster processing
                yolo_model = None
                yolo_model_load_error = None
                if enable_body_pose:
                    if not _HAS_ULTRALYTICS:
                        yolo_model_load_error = "ultralytics_not_installed"
                    else:
                        try:
                            yolo_model = _load_yolo_model(yolo_weights_path)
                        except Exception as e:
                            yolo_model_load_error = str(e)
                            yolo_model = None

                keypoint_data = {
                    "metadata": {
                        "filename": uploaded_file.name,
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "total_frames": total_frames,
                        "hands": {
                            "backend": "mediapipe",
                            "model_complexity": model_complexity,
                            "min_detection_confidence": min_detection_confidence,
                            "min_tracking_confidence": min_tracking_confidence,
                        },
                        "body_pose": {
                            "enabled": bool(enable_body_pose),
                            "backend": "yolo" if enable_body_pose else None,
                            "weights": yolo_weights_path if enable_body_pose else None,
                            "conf": yolo_conf if enable_body_pose else None,
                            "iou": yolo_iou if enable_body_pose else None,
                            "model_load_error": yolo_model_load_error,
                        },
                    },
                    "frames": [],
                }

                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()

                # MediaPipe Hands instance for file processing (single-threaded here)
                with mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    model_complexity=model_complexity,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence,
                ) as hands_processor:

                    frame_idx = 0
                    while cap.isOpened():
                        success, image = cap.read()
                        if not success:
                            break

                        # --- Hands (always) ---
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = hands_processor.process(image_rgb)

                        hands_out = []
                        if results.multi_hand_landmarks and results.multi_handedness:
                            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                                mp_drawing.draw_landmarks(
                                    image,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style(),
                                )

                                hand_info = {
                                    "label": handedness.classification[0].label,
                                    "score": float(handedness.classification[0].score),
                                    "landmarks": [
                                        {
                                            "x": float(lm.x),
                                            "y": float(lm.y),
                                            "z": float(lm.z),
                                            "visibility": float(lm.visibility) if hasattr(lm, "visibility") else 1.0,
                                        }
                                        for lm in hand_landmarks.landmark
                                    ],
                                }
                                hands_out.append(hand_info)

                        # --- Body pose (optional) ---
                        pose_persons = []
                        pose_error = None
                        if enable_body_pose:
                            if yolo_model is None:
                                pose_error = yolo_model_load_error or "yolo_model_not_loaded"
                            else:
                                try:
                                    pose_persons = _run_yolo_pose_on_frame(image, yolo_model)
                                except Exception as e:
                                    pose_error = str(e)
                                    pose_persons = []

                        frame_data = {
                            "frame_index": frame_idx,
                            "timestamp_sec": frame_idx / fps if fps and fps > 0 else 0,
                            "hands": hands_out,
                            "body_pose": {
                                "enabled": bool(enable_body_pose),
                                "persons": pose_persons,
                                "error": pose_error,
                            },
                        }

                        keypoint_data["frames"].append(frame_data)
                        out.write(image)

                        frame_idx += 1

                        if frame_idx % max(1, (total_frames // 100)) == 0:
                            progress = min(frame_idx / total_frames, 1.0)
                            progress_bar.progress(progress)
                            elapsed = time.time() - start_time
                            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
                            status_text.text(f"Processed {frame_idx}/{total_frames} frames ( {fps_proc:.2f} fps )")

                cap.release()
                out.release()
                progress_bar.progress(1.0)
                status_text.text("Processing complete! Preparing outputs...")

                with open(output_json_path, "w") as f:
                    json.dump(keypoint_data, f, indent=2)

                st.success("Analysis Complete!")

                st.markdown("### Processed Video Result")
                st.video(output_video_path)

                col1, col2 = st.columns(2)

                with col1:
                    with open(output_video_path, "rb") as f:
                        st.download_button(
                            label="Download Annotated Video (MP4)",
                            data=f,
                            file_name=f"annotated_{uploaded_file.name}",
                            mime="video/mp4",
                        )

                with col2:
                    with open(output_json_path, "r") as f:
                        st.download_button(
                            label="Download Keypoint Data (JSON)",
                            data=f,
                            file_name=f"keypoints_{uploaded_file.name}.json",
                            mime="application/json",
                        )

                # Cleanup temp files
                try:
                    os.remove(input_video_path)
                    os.remove(output_video_path)
                    os.remove(output_json_path)
                except Exception:
                    pass
