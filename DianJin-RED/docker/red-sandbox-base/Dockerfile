ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/root
ENV PATH="/usr/local/bin:/root/.local/bin:${PATH}"

RUN mkdir -p /root/.pip \
    && printf "[global]\nindex-url = https://pypi.org/simple\n" > /root/.pip/pip.conf

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        jq \
        procps \
        python3 \
        python3-pip \
        python3-venv \
        ripgrep \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /workspace /workspace/.world_state /app

WORKDIR /workspace

CMD ["sleep", "infinity"]
