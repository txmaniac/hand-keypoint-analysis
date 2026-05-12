# Use NVIDIA's official PyTorch container (PyTorch 2.1.0, CUDA 12.2, Ubuntu 22.04)
# This perfectly matches your compute server's hardware constraints
FROM nvcr.io/nvidia/pytorch:23.10-py3

# Set the working directory inside the container
WORKDIR /app

# Install system-level OpenGL libraries just in case downstream packages 
# ever attempt to bypass our headless configuration in the future.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy our minimal requirements file for the headless script
COPY req-test.txt .

# ---------------------------------------------------------
# DEPENDENCY HELL RESOLUTION PIPELINE
# ---------------------------------------------------------
# 1. Install our requirements (Ultralytics & Mediapipe will sneak GUI OpenCVs in here)
RUN pip install --no-cache-dir -r req-test.txt

# 2. Rip out the corrupting GUI packages
RUN pip uninstall -y opencv-python opencv-contrib-python

# 3. Force reinstall MediaPipe to fix the "missing solutions" broken wheel issue, 
# and lock in the highly stable 4.7.x Headless OpenCVs to prevent DictValue crashes.
RUN pip install --force-reinstall --no-cache-dir mediapipe>=0.10.0 && \
    pip install --no-cache-dir opencv-python-headless==4.7.0.72 opencv-contrib-python-headless==4.7.0.72

# Copy the headless tracking script
COPY run_yolo_headless.py .

# Expose the entrypoint so you can run the container by mapping volumes
# Example usage:
# docker run --gpus all -v /local/videos:/app/input -v /local/results:/app/output hand-tracker
ENTRYPOINT ["python", "run_yolo_headless.py"]
CMD ["/app/input", "/app/output", "--hands", "Both"]
