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
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, WebRtcMode

# -----------------------------
# Initialize MediaPipe
# -----------------------------
mp_holistic = mp.solutions.holistic
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

st.set_page_config(page_title="Holistic Analyics Engine", page_icon="📈", layout="wide")

st.title("Holistic Keypoint Analytics System")
st.markdown(
    "Analyze full body dynamics, hand interactions, and compare trajectories across multiple elicitation video studies."
)

# -----------------------------
# Sidebar configuration
# -----------------------------
st.sidebar.header("Configuration")

# Toggle to bypass holistic model processing entirely
enable_body_pose = st.sidebar.toggle(
    "Enable Body Pose", 
    value=True, 
    help="When turned off, the system will use the lighter, faster `Hands` model to purely capture finger interactions."
)

model_complexity = st.sidebar.selectbox("Model Complexity", [0, 1, 2], index=1, help="0 is fastest, 2 is most accurate but slowest (1 is baseline).")
min_detection_confidence = st.sidebar.slider("Min Detection Confidence", 0.0, 1.0, 0.7)
min_tracking_confidence = st.sidebar.slider("Min Tracking Confidence", 0.0, 1.0, 0.7)

tab1, tab2, tab3 = st.tabs(["Upload Video", "Live Webcam", "Comparative Analytics Dashboard"])

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

def _analyze_frame(image_bgr):
    """Annotate frame with selected MediaPipe model dynamically."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    pose_data = []
    left_hand = []
    right_hand = []

    if enable_body_pose:
        if not hasattr(thread_local, "holistic"):
            thread_local.holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                refine_face_landmarks=False # Disabled for performance and file size
            )

        results = thread_local.holistic.process(img_rgb)
        
        # Plotting
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

        pose_data = serialize_landmarks(results.pose_landmarks)
        left_hand = serialize_landmarks(results.left_hand_landmarks)
        right_hand = serialize_landmarks(results.right_hand_landmarks)
        
    else:
        # Fallback to the purely lightweight Hands model
        if not hasattr(thread_local, "hands"):
            thread_local.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=model_complexity if model_complexity < 2 else 1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            
        results = thread_local.hands.process(img_rgb)
        
        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(
                    image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )
                if handedness.classification[0].label == 'Left':
                   left_hand = serialize_landmarks(hand_landmarks)
                else:
                   right_hand = serialize_landmarks(hand_landmarks)

    return {
        "pose": pose_data,
        "left_hand": left_hand,
        "right_hand": right_hand
    }

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    _ = _analyze_frame(img)
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
        async_processing=False,
    )

# -----------------------------
# Upload video tab
# -----------------------------
with tab1:
    uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "webm"])

    if uploaded_file is not None:
        if st.button("Start File Analysis"):
            with st.spinner("Executing Tracking Engine..."):
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile.write(uploaded_file.read())
                tfile.close()
                input_video_path = tfile.name

                output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                output_json_path = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name

                cap = cv2.VideoCapture(input_video_path)

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                raw_out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
                out = cv2.VideoWriter(raw_out_path, fourcc, fps, (width, height))

                keypoint_data = {
                    "metadata": {
                        "filename": uploaded_file.name,
                        "fps": fps,
                        "total_frames": total_frames,
                    },
                    "frames": [],
                }

                progress_bar = st.progress(0)
                status_text = st.empty()
                start_time = time.time()
                
                # Context Management Logic for Model Swap
                engine_config = None
                if enable_body_pose:
                    engine_config = mp_holistic.Holistic(
                        static_image_mode=False,
                        model_complexity=model_complexity,
                        min_detection_confidence=min_detection_confidence,
                        min_tracking_confidence=min_tracking_confidence,
                        refine_face_landmarks=False
                    )
                else:
                    engine_config = mp_hands.Hands(
                         static_image_mode=False,
                         max_num_hands=2,
                         model_complexity=model_complexity if model_complexity < 2 else 1,
                         min_detection_confidence=min_detection_confidence,
                         min_tracking_confidence=min_tracking_confidence
                    )

                with engine_config as tracker:
                    frame_idx = 0
                    while cap.isOpened():
                        success, image = cap.read()
                        if not success:
                            break

                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = tracker.process(image_rgb)
                        
                        pose_data = []
                        left_hand = []
                        right_hand = []

                        if enable_body_pose:
                            if results.pose_landmarks:
                                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                                pose_data = serialize_landmarks(results.pose_landmarks)
                            if results.left_hand_landmarks:
                                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
                                left_hand = serialize_landmarks(results.left_hand_landmarks)
                            if results.right_hand_landmarks:
                                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
                                right_hand = serialize_landmarks(results.right_hand_landmarks)
                        else:
                            if results.multi_hand_landmarks and results.multi_handedness:
                                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                                    mp_drawing.draw_landmarks(
                                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                                    )
                                    if handedness.classification[0].label == 'Left':
                                       left_hand = serialize_landmarks(hand_landmarks)
                                    else:
                                       right_hand = serialize_landmarks(hand_landmarks)
                                       
                        frame_data = {
                            "frame_index": frame_idx,
                            "timestamp_sec": frame_idx / fps,
                            "pose": pose_data,
                            "left_hand": left_hand,
                            "right_hand": right_hand
                        }

                        keypoint_data["frames"].append(frame_data)
                        out.write(image)

                        frame_idx += 1
                        if frame_idx % max(1, (total_frames // 100)) == 0:
                            progress_bar.progress(min(frame_idx / total_frames, 1.0))
                            status_text.text(f"Processed {frame_idx}/{total_frames} frames ( {(frame_idx / (time.time() - start_time)):.2f} fps )")

                cap.release()
                out.release()
                
                # Execute FFmpeg securely to convert the raw linux mp4 into Web-safe H264
                status_text.text("Converting video to Web-safe format... Please wait.")
                os.system(f"ffmpeg -y -i {raw_out_path} -vcodec libx264 -f mp4 {output_video_path}")
                
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")

                with open(output_json_path, "w") as f:
                    json.dump(keypoint_data, f, indent=2)

                st.video(output_video_path)
                
                col1, col2 = st.columns(2)
                with col1:
                    with open(output_video_path, "rb") as f:
                        st.download_button("Download Annotated Video (MP4)", data=f, file_name=f"processed_{uploaded_file.name}", mime="video/mp4")
                with col2:
                    with open(output_json_path, "r") as f:
                        st.download_button("Download Keypoint Data (JSON)", data=f, file_name=f"processed_{uploaded_file.name.replace('.mp4','.json')}", mime="application/json")


# -----------------------------
# Analytics Dashboard Tab
# -----------------------------
with tab3:
    st.markdown("### 📊 Comparative Analysis Dashboard")
    st.info("Upload multiple generated `.json` keypoint files to compare biomechanical tracks side-by-side.")
    
    comparative_files = st.file_uploader("Upload Output JSON datasets", type=["json"], accept_multiple_files=True)
    
    if comparative_files:
        st.divider()
        dfs_distance = []
        dfs_visibility = []
        
        for file in comparative_files:
            data = json.load(file)
            fname = data["metadata"]["filename"]
            fps = data["metadata"]["fps"]
            
            times = []
            distances = []
            l_vis = []
            r_vis = []
            
            for frame in data["frames"]:
                t = frame["timestamp_sec"]
                pose = frame.get("pose", [])
                
                # Wrist landmarks in MediaPipe Pose: Left=15, Right=16
                distance = np.nan
                if len(pose) > 16:
                    lx, ly = pose[15]["x"], pose[15]["y"]
                    rx, ry = pose[16]["x"], pose[16]["y"]
                    distance = np.sqrt((rx - lx)**2 + (ry - ly)**2)
                
                # Check Hand tracking presence
                lv = 1 if len(frame.get("left_hand", [])) > 0 else 0
                rv = 1 if len(frame.get("right_hand", [])) > 0 else 0
                
                times.append(t)
                distances.append(distance)
                l_vis.append(lv)
                r_vis.append(rv)
                
            df_dist = pd.DataFrame({"Time (s)": times, "Wrist Distance": distances, "Video": fname})
            df_vis = pd.DataFrame({"Time (s)": times, "Left Hand Presence": l_vis, "Right Hand Presence": r_vis, "Video": fname})
            
            dfs_distance.append(df_dist)
            dfs_visibility.append(df_vis)
            
        if dfs_distance:
            final_dist = pd.concat(dfs_distance)
            final_vis = pd.concat(dfs_visibility)
            
            if not final_dist["Wrist Distance"].isna().all():
                st.markdown("#### Distance Between Wrists (Extension proxy)")
                fig1 = px.line(final_dist, x="Time (s)", y="Wrist Distance", color="Video", title="Wrist Extension Dynamics over Time")
                st.plotly_chart(fig1, use_container_width=True)
            else:
                 st.warning("Wrist Distance charts are unavailable because Body Pose was completely disabled in the uploaded files.")
                 
            st.markdown("#### Left Hand Tracking Visibility %")
            final_vis["Left Hand Density"] = final_vis.groupby("Video")["Left Hand Presence"].transform(lambda x: x.rolling(10, min_periods=1).mean())
            fig2 = px.line(final_vis, x="Time (s)", y="Left Hand Density", color="Video", title="Left Hand Activity Window")
            st.plotly_chart(fig2, use_container_width=True)
            
            st.markdown("#### Right Hand Tracking Visibility %")
            final_vis["Right Hand Density"] = final_vis.groupby("Video")["Right Hand Presence"].transform(lambda x: x.rolling(10, min_periods=1).mean())
            fig3 = px.line(final_vis, x="Time (s)", y="Right Hand Density", color="Video", title="Right Hand Activity Window")
            st.plotly_chart(fig3, use_container_width=True)
