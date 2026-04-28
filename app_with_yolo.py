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
        async_processing=True,
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

                output_container = av.open(output_video_path, mode='w', format='mp4', options={'movflags': 'faststart'})
                output_stream = output_container.add_stream('libx264', rate=int(fps))
                output_stream.width = width
                output_stream.height = height
                output_stream.pix_fmt = 'yuv420p'

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
                        frame_out = av.VideoFrame.from_ndarray(image, format='bgr24')
                        for packet in output_stream.encode(frame_out):
                            output_container.mux(packet)

                        frame_idx += 1
                        if frame_idx % max(1, (total_frames // 100)) == 0:
                            progress_bar.progress(min(frame_idx / total_frames, 1.0))
                            status_text.text(f"Processed {frame_idx}/{total_frames} frames ( {(frame_idx / (time.time() - start_time)):.2f} fps )")

                cap.release()
                
                # Flush the PyAV encoder
                for packet in output_stream.encode():
                    output_container.mux(packet)
                output_container.close()
                
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")

                with open(output_json_path, "w") as f:
                    json.dump(keypoint_data, f, indent=2)

                st.session_state['processed_video_path'] = output_video_path
                st.session_state['processed_json_path'] = output_json_path
                st.session_state['processed_file_name'] = uploaded_file.name

        if st.session_state.get('processed_file_name') == uploaded_file.name:
            out_vid = st.session_state['processed_video_path']
            out_json = st.session_state['processed_json_path']
            
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
        
        for file in comparative_files:
            data = json.load(file)
            fname = data["metadata"]["filename"]
            fps = data["metadata"]["fps"]
            
            records = []
            
            for frame in data["frames"]:
                t = frame["timestamp_sec"]
                pose = frame.get("pose", [])
                lh = frame.get("left_hand", [])
                rh = frame.get("right_hand", [])
                
                # Pose Wrist Distance (fallback)
                pose_dist = np.nan
                if len(pose) > 16:
                    pose_dist = np.sqrt((pose[15]["x"] - pose[16]["x"])**2 + (pose[15]["y"] - pose[16]["y"])**2)
                
                l_vis = 1 if len(lh) > 0 else 0
                r_vis = 1 if len(rh) > 0 else 0
                
                def dist3d(a, b):
                    return np.sqrt((a['x']-b['x'])**2 + (a['y']-b['y'])**2 + (a['z']-b['z'])**2)
                
                inter_hand_dist = dist3d(lh[0], rh[0]) if l_vis and r_vis else np.nan
                
                l_pinch = dist3d(lh[4], lh[8]) if l_vis and len(lh) > 8 else np.nan
                r_pinch = dist3d(rh[4], rh[8]) if r_vis and len(rh) > 8 else np.nan
                
                l_open = dist3d(lh[0], lh[12]) if l_vis and len(lh) > 12 else np.nan
                r_open = dist3d(rh[0], rh[12]) if r_vis and len(rh) > 12 else np.nan
                
                records.append({
                    "Time (s)": t,
                    "Video": fname,
                    "Pose Wrist Distance": pose_dist,
                    "Left Hand Presence": l_vis,
                    "Right Hand Presence": r_vis,
                    "Inter-Hand Distance": inter_hand_dist,
                    "Left Pinch": l_pinch,
                    "Right Pinch": r_pinch,
                    "Left Openness": l_open,
                    "Right Openness": r_open,
                    "L_Wx": lh[0]['x'] if l_vis else np.nan,
                    "L_Wy": lh[0]['y'] if l_vis else np.nan,
                    "L_Wz": lh[0]['z'] if l_vis else np.nan,
                    "R_Wx": rh[0]['x'] if r_vis else np.nan,
                    "R_Wy": rh[0]['y'] if r_vis else np.nan,
                    "R_Wz": rh[0]['z'] if r_vis else np.nan,
                })
            
            df = pd.DataFrame(records)
            
            # Velocity Calculation
            df['dt'] = df['Time (s)'].diff().fillna(1.0/fps)
            df['L_Vel'] = np.sqrt(df['L_Wx'].diff()**2 + df['L_Wy'].diff()**2 + df['L_Wz'].diff()**2) / df['dt']
            df['R_Vel'] = np.sqrt(df['R_Wx'].diff()**2 + df['R_Wy'].diff()**2 + df['R_Wz'].diff()**2) / df['dt']
            
            # Tremor (Smoothness) Calculation
            window_size = max(3, int(0.5 * fps))
            df['L_Tremor'] = df['L_Vel'].rolling(window=window_size, min_periods=1).std()
            df['R_Tremor'] = df['R_Vel'].rolling(window=window_size, min_periods=1).std()
            
            dfs_all.append(df)
            
        if dfs_all:
            final_df = pd.concat(dfs_all)
            
            st.markdown("#### 1. Bimanual Coordination")
            fig_inter = px.line(final_df, x="Time (s)", y="Inter-Hand Distance", color="Video", title="Distance Between Left and Right Hand")
            st.plotly_chart(fig_inter, use_container_width=True)
            
            st.markdown("#### 2. Activity State (Moving vs Resting)")
            rest_thresh = st.slider("Resting Threshold (speed in normalized units/sec)", 0.0, 2.0, 0.1, 0.05, help="Hand speeds below this value are classified as 'Resting'.")
            
            final_df['L_State'] = np.where(final_df['L_Vel'] < rest_thresh, 'Resting', 'Moving')
            final_df['R_State'] = np.where(final_df['R_Vel'] < rest_thresh, 'Resting', 'Moving')
            
            col_l, col_r = st.columns(2)
            
            with col_l:
                state_counts_l = final_df.dropna(subset=['L_Vel']).groupby(['Video', 'L_State']).size().reset_index(name='Frames')
                fig_state_l = px.bar(state_counts_l, x="Video", y="Frames", color="L_State", title="Left Hand Activity Proportion", barmode='stack')
                st.plotly_chart(fig_state_l, use_container_width=True)
                
                fig_tremor_l = px.line(final_df, x="Time (s)", y="L_Tremor", color="Video", title="Left Movement Jitter (Tremor Proxy)")
                st.plotly_chart(fig_tremor_l, use_container_width=True)
                
                fig_pinch_l = px.line(final_df, x="Time (s)", y="Left Pinch", color="Video", title="Left Pinch Grip Distance")
                st.plotly_chart(fig_pinch_l, use_container_width=True)
                
                fig_open_l = px.line(final_df, x="Time (s)", y="Left Openness", color="Video", title="Left Hand Openness (Wrist to Middle Tip)")
                st.plotly_chart(fig_open_l, use_container_width=True)

            with col_r:
                state_counts_r = final_df.dropna(subset=['R_Vel']).groupby(['Video', 'R_State']).size().reset_index(name='Frames')
                fig_state_r = px.bar(state_counts_r, x="Video", y="Frames", color="R_State", title="Right Hand Activity Proportion", barmode='stack')
                st.plotly_chart(fig_state_r, use_container_width=True)
                
                fig_tremor_r = px.line(final_df, x="Time (s)", y="R_Tremor", color="Video", title="Right Movement Jitter (Tremor Proxy)")
                st.plotly_chart(fig_tremor_r, use_container_width=True)
                
                fig_pinch_r = px.line(final_df, x="Time (s)", y="Right Pinch", color="Video", title="Right Pinch Grip Distance")
                st.plotly_chart(fig_pinch_r, use_container_width=True)
                
                fig_open_r = px.line(final_df, x="Time (s)", y="Right Openness", color="Video", title="Right Hand Openness (Wrist to Middle Tip)")
                st.plotly_chart(fig_open_r, use_container_width=True)

            if not final_df["Pose Wrist Distance"].isna().all():
                st.markdown("#### Pose Tracking (Fallback)")
                fig_pose = px.line(final_df, x="Time (s)", y="Pose Wrist Distance", color="Video", title="Wrist Distance (from Pose framework)")
                st.plotly_chart(fig_pose, use_container_width=True)
