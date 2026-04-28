# Running Command
`streamlit run app_with_yolo.py`

# Analytics Docs
# Comparative Analytics Dashboard Enhancement

### 1. Inter-Hand Distance
We will calculate the 3D Euclidean distance between the left wrist (Landmark 0) and the right wrist (Landmark 0) over time. This provides insight into bimanual coordination (how closely the hands are working together). We will visualize this as a continuous line chart.

### 2. Moving vs. Resting States (Velocity & Activation)
To determine if a hand is moving or resting:
- We will compute the inter-frame velocity (speed) of each wrist by measuring the change in 3D position over the time delta.
- We will apply a speed threshold. If the speed is below the threshold, the hand is classified as "Resting"; otherwise, it is "Moving".
- We will visualize this using an **Activity Timeline** (showing moving vs resting states over time) and an **Activity Proportion** chart (e.g., 70% moving, 30% resting).

### 3. Wrist & Fine Motor Mechanics
We will add standard metrics from hand keypoint literature:
- **Pinch Grip Distance**: The distance between the Thumb Tip (Landmark 4) and the Index Finger Tip (Landmark 8). This is a critical metric for grasping and precision tasks.
- **Hand Openness / Extension**: The distance between the Wrist (Landmark 0) and the Middle Finger Tip (Landmark 12). This acts as a proxy for how open or clenched the fist is.

### 4. Movement Smoothness / Tremor Proxy
To measure the stability of the hands:
- We will calculate the "Jitter" or Movement Smoothness by taking the rolling standard deviation of the wrist's velocity.
- High values indicate jerky or tremorous movements, while low values indicate smooth, controlled trajectories.
