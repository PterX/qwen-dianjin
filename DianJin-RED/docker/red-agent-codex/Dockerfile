ARG BASE_IMAGE=red-sandbox-base:v1
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

ENV NODE_ENV=production
ARG CODEX_VERSION=0.142.5
ARG NPM_REGISTRY=https://registry.npmjs.org

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm config set registry "${NPM_REGISTRY}" \
    && npm install -g "@openai/codex@${CODEX_VERSION}" \
    && codex --version

LABEL red-agent-world.runtime="codex"

WORKDIR /workspace
