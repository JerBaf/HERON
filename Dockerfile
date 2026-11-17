# HERON (NeurIPS 2026) training image.
# PyTorch with CUDA 12.6 on a slim Python base. The GPU driver comes from the host through the NVIDIA Container
# Toolkit (docker run --gpus all), so the image runs with any driver recent enough for CUDA 12.6.

# ---- build stage - install the pinned environment into a virtualenv ----
FROM python:3.12-slim AS builder
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-cache-dir --no-deps \
        --index-url https://download.pytorch.org/whl/cu126 \
        --extra-index-url https://pypi.org/simple \
        -r /tmp/requirements.lock
COPY . /src
RUN pip install --no-cache-dir --no-deps --no-build-isolation /src

# ---- runtime stage ----
FROM python:3.11-slim
RUN useradd --create-home heron
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=all
WORKDIR /app
COPY configs/ configs/
COPY examples/ examples/
COPY scripts/ scripts/
COPY docker/ docker/
USER heron
ENTRYPOINT ["/bin/sh", "-c", "/app/docker/entrypoint.sh"]
CMD ["network"]
