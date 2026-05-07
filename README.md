# Holistic Keypoint Analytics Engine

A robust, high-fidelity hand tracking and biomechanical analytics pipeline. Designed to extract incredibly precise hand kinematics from video footage—even under conditions of heavy motion blur, awkward camera angles (POV/Top-down), and partial occlusion.

## Quickstart

Run the real-time processing and analytics dashboard:
```bash
streamlit run app_with_yolo.py
```

Run the headless batch-processor (for compute servers):
```bash
python run_yolo_headless.py --input_dir ./videos --output_dir ./results
```

## Pipeline Architecture (V2)

The engine utilizes a **Two-Stage Hybrid Architecture**, bridging the robust context-awareness of YOLOv8 with the high-precision local feature extraction of MediaPipe Hands. To ensure maximum data continuity, the pipeline implements a 4-tier fallback system.

### Tier 1: Primary Full-Frame Scan (High-Confidence)
The first pass scans the entire video frame using MediaPipe Hands with a strict `0.5` confidence threshold. This high threshold guarantees that background textures or random objects are not hallucinated as hands. 

### Tier 2: YOLO Kinematic ROI Fallback (High-Recall)
If Tier 1 fails to detect a hand (due to motion blur, rapid movement, or occlusion), the system activates the **Dual-Threshold Engine**:
1. **YOLO Pose Extraction**: A lightweight YOLOv8 Pose model (`yolo26n-pose.pt`) extracts the body pose. Because YOLO is highly occlusion-resistant, it easily identifies the wrist and elbow even when the fingers are hidden.
2. **Conditional Kinematics**: 
   - If the elbow is visible, the system calculates the forearm vector and projects it forward to mathematically estimate the center of the palm, scaling the crop box dynamically based on the forearm length.
   - If the elbow is not visible (common in POV/Top-down footage), the system falls back to a slightly larger crop box centered directly on the wrist.
3. **Multi-Scale Low-Confidence Extraction**: The cropped Region of Interest (ROI) is passed to a *second*, dedicated MediaPipe Hands model. Because YOLO has already verified that the crop contains a wrist, the confidence threshold on this secondary model is drastically lowered to `0.15`. This forces the network to aggressively capture blurry or partially occluded hands. If the initial crop fails, the system instantly attempts multi-scale cropping (zooming in and out) to ensure a perfect fit.
4. **Strict Handedness Enforcement**: To prevent the model from accidentally locking onto the wrong hand when arms cross, the crop scanner explicitly filters out hands that do not match the expected handedness label.

### Tier 3: Kalman Filter Prediction (Momentum Tracking)
If both neural networks fail, a physical **Kalman Filter** takes over. Using a constant-velocity model with built-in friction dampening, the filter calculates the trajectory, velocity, and momentum of the hand prior to it disappearing. It predicts the spatial coordinates of all 21 keypoints for up to 10 frames, allowing the hand to "coast" smoothly to a stop when occluded.

### Tier 4: Scipy Linear Interpolation (Bridging Gaps)
Once the hand is re-detected by the neural networks, a post-processing pass uses `scipy.interpolate.interp1d` to bridge the gap between the Kalman predicted coordinates and the newly detected coordinates, ensuring a 100% continuous, fluid trajectory across the entire video.

---

## Comparative Analytics Dashboard

Once processing is complete, the exported JSON data can be loaded into the dashboard to visualize complex biomechanical metrics.

### 1. Bimanual Coordination
Calculates the 3D Euclidean distance between the left wrist and the right wrist over time. This provides insight into how closely the hands are working together during a task.

### 2. Fine Motor Mechanics (Pinch Grip)
Measures the precise 3D distance between the Thumb Tip (Landmark 4) and the Index Finger Tip (Landmark 8). This is a critical metric for evaluating grasping ability and precision motor control.

### 3. Movement Smoothness / Tremor Proxy
Calculates the "Jitter" by taking the rolling standard deviation of the wrist's inter-frame velocity. 
- **High values** indicate jerky, tremorous, or uncoordinated movements.
- **Low values** indicate smooth, highly controlled trajectories.

### 4. Source Reliability Heatmap
A comprehensive visual timeline that color-codes exactly how each frame's keypoints were generated:
- 🟢 **Green (Directly Detected)**: Found by the Tier 1 primary scan.
- 🟡 **Yellow (ROI Fallback)**: Rescued by the Tier 2 YOLO low-confidence crop.
- 🟠 **Orange (Kalman Predicted)**: Physically simulated by the Tier 3 filter.
- 🟣 **Magenta (Interpolated)**: Mathematically bridged by the Tier 4 interpolator.
- 🔴 **Red (Missing)**: Total loss of tracking.
