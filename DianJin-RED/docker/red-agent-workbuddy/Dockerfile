ARG BASE_IMAGE=red-sandbox-base:v1
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

ENV NODE_ENV=production
ARG WORKBUDDY_VERSION=2.133.1

RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get update \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install CodeBuddy Code CLI (Tencent WorkBuddy)
RUN npm config set registry https://registry.npmjs.org \
    && npm install -g "@tencent-ai/codebuddy-code@${WORKBUDDY_VERSION}" \
    && codebuddy --version

LABEL red-agent-world.runtime="workbuddy"

WORKDIR /workspace
