ARG BASE_IMAGE=red-sandbox-base:v1
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

ENV NODE_ENV=production
ARG OPENCLAW_VERSION=2026.7.1-2

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm config set registry https://registry.npmjs.org \
    && npm install -g "openclaw@${OPENCLAW_VERSION}" \
    && openclaw --version

LABEL red-agent-world.runtime="openclaw"

WORKDIR /workspace
