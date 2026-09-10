FROM cyberbotics/webots:R2025a-ubuntu22.04

USER root
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends python3-pip \
    && python3 -m pip install --no-cache-dir \
       "numpy>=1.26,<3" "onnxruntime>=1.19,<2" "Pillow>=10,<12" \
    && rm -rf /var/lib/apt/lists/*
