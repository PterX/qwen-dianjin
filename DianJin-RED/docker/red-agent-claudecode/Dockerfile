ARG BASE_IMAGE=red-sandbox-base:v1
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

ENV NODE_ENV=production
ARG CLAUDE_CODE_VERSION=2.1.227

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

RUN npm config set registry https://registry.npmjs.org \
    && npm install -g "@anthropic-ai/claude-code@${CLAUDE_CODE_VERSION}" \
    && claude --version

LABEL red-agent-world.runtime="claudecode"

WORKDIR /workspace
