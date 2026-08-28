ARG BASE_IMAGE=red-sandbox-base:v1
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

ENV HERMES_HOME=/root/.hermes
ARG HERMES_REF=9829746dfe3d5077f4f076e257505ec7f8feaa65

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libffi-dev \
        python3-dev \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN set -euo pipefail \
    && git clone https://github.com/NousResearch/hermes-agent.git /opt/hermes-agent-src \
    && cd /opt/hermes-agent-src \
    && git checkout "$HERMES_REF" \
    && python3 -m venv /opt/hermes-agent-venv \
    && /opt/hermes-agent-venv/bin/pip install --no-cache-dir --index-url https://pypi.org/simple ".[mcp,cli]" edge-tts==7.2.7 \
    && ln -sf /opt/hermes-agent-venv/bin/hermes /usr/local/bin/hermes \
    && hermes --version

LABEL red-agent-world.runtime="hermes"

WORKDIR /workspace
