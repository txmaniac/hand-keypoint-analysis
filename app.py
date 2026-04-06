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

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

st.set_page_config(page_title="Hand Keypoint Analysis", page_icon="🖐️", layout="wide")

st.title("Hand Keypoint Analysis System")
st.markdown("Analyze hand keypoints via uploaded video files or live webcam stream with 21-point accurate finger tracking.")

st.sidebar.header("Configuration")
model_complexity = st.sidebar.selectbox("Model Complexity", [0, 1], index=0, help="0 is a smaller, faster model. 1 is larger but slower.")
min_detection_confidence = st.sidebar.slider("Min Detection Confidence", 0.0, 1.0, 0.8)
min_tracking_confidence = st.sidebar.slider("Min Tracking Confidence", 0.0, 1.0, 0.8)

tab1, tab2 = st.tabs(["Upload Video", "Live Webcam Analysis"])

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

thread_local = threading.local()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Lazily initialize MediaPipe on the precise dedicated thread spawned by WebRTC.
    # This prevents the thread-desync freezes that MediaPipe's C++ bindings suffer from.
    if not hasattr(thread_local, "hands"):
        thread_local.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
    results = thread_local.hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    return av.VideoFrame.from_ndarray(img, format="bgr24")

with tab2:
    st.markdown("### Webcam Stream")
    webrtc_streamer(
        key="hand-tracking",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False # MediaPipe needs synchronous execution to prevent dropping/freezing frames
    )

with tab1:
    uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "webm"])

    if uploaded_file is not None:
        st.markdown("### Uploaded Video")
        st.video(uploaded_file)
        
        if st.button("Start File Analysis"):
            with st.spinner("Processing video... This may take a while depending on the video length and resolution."):
                
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.read())
                input_video_path = tfile.name
                
                output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                output_json_path = tempfile.NamedTemporaryFile(delete=False, suffix='.json').name
                
                cap = cv2.VideoCapture(input_video_path)
                
                width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps    = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                if total_frames <= 0:
                    st.error("Could not read video frames. The file might be corrupted.")
                    st.stop()
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                
                keypoint_data = {
                    "metadata": {
                        "filename": uploaded_file.name,
                        "width": width,
                        "height": height,
                        "fps": fps,
                        "total_frames": total_frames
                    },
                    "frames": []
                }
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                start_time = time.time()
                
                with mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    model_complexity=model_complexity,
                    min_detection_confidence=min_detection_confidence,
                    min_tracking_confidence=min_tracking_confidence) as hands:
                    
                    frame_idx = 0
                    while cap.isOpened():
                        success, image = cap.read()
                        if not success:
                            break
                            
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        results = hands.process(image_rgb)
                        
                        frame_data = {
                            "frame_index": frame_idx,
                            "timestamp_sec": frame_idx / fps if fps > 0 else 0,
                            "hands": []
                        }
                        
                        if results.multi_hand_landmarks and results.multi_handedness:
                            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                                mp_drawing.draw_landmarks(
                                    image,
                                    hand_landmarks,
                                    mp_hands.HAND_CONNECTIONS,
                                    mp_drawing_styles.get_default_hand_landmarks_style(),
                                    mp_drawing_styles.get_default_hand_connections_style())
                                    
                                hand_info = {
                                    "label": handedness.classification[0].label,
                                    "score": handedness.classification[0].score,
                                    "landmarks": []
                                }
                                
                                for lm in hand_landmarks.landmark:
                                    hand_info["landmarks"].append({
                                        "x": lm.x,
                                        "y": lm.y,
                                        "z": lm.z,
                                        "visibility": lm.visibility if hasattr(lm, 'visibility') else 1.0
                                    })
                                
                                frame_data["hands"].append(hand_info)
                                
                        keypoint_data["frames"].append(frame_data)
                        out.write(image)
                        
                        frame_idx += 1
                        
                        if frame_idx % max(1, (total_frames // 100)) == 0:
                            progress = min(frame_idx / total_frames, 1.0)
                            progress_bar.progress(progress)
                            elapsed = time.time() - start_time
                            fps_proc = frame_idx / elapsed
                            status_text.text(f"Processed {frame_idx}/{total_frames} frames ( {fps_proc:.2f} fps )")

                cap.release()
                out.release()
                progress_bar.progress(1.0)
                status_text.text("Processing complete! Preparing outputs...")
                
                web_ready_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                os.system(f"ffmpeg -y -i {output_video_path} -vcodec libx264 -f mp4 {web_ready_video_path}")
                
                with open(output_json_path, 'w') as f:
                    json.dump(keypoint_data, f, indent=4)
                
                st.success("Analysis Complete!")
                
                st.markdown("### Processed Video Result")
                st.video(web_ready_video_path)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    with open(web_ready_video_path, 'rb') as f:
                        st.download_button(
                            label="Download Annotated Video (MP4)",
                            data=f,
                            file_name=f"annotated_{uploaded_file.name}",
                            mime="video/mp4"
                        )
                
                with col2:
                    with open(output_json_path, 'r') as f:
                        st.download_button(
                            label="Download Keypoint Data (JSON)",
                            data=f,
                            file_name=f"keypoints_{uploaded_file.name}.json",
                            mime="application/json"
                        )

                try:
                    os.remove(input_video_path)
                    os.remove(output_video_path)
                except:
                    pass
