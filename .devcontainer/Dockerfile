# Use official NVIDIA CUDA development image as the base
FROM nvidia/cuda:12.6.0-devel-ubuntu24.04

# Avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install uv early so we can provision the exact Python runtime version.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Install system dependencies, Node.js, Python, Git, and sudo.
# sudo is included so AI agents inside the container can install additional
# software at runtime. This only affects the container's own filesystem layers,
# never the host OS (the only host-visible path is the workspace bind mount).
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ca-certificates \
    sudo \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && uv python install 3.10.20 \
    && PYBIN="$(uv python find 3.10.20)" \
    && ln -sf "$PYBIN" /usr/local/bin/python3 \
    && ln -sf "$PYBIN" /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Remove the default 'ubuntu' user (UID 1000) so we can reuse that UID
RUN userdel -r ubuntu || true

# Create a non-root user "developer" (UID 1000) with passwordless sudo so agents
# can install software inside the container without being blocked.
RUN useradd -m -s /bin/bash -u 1000 developer \
    && echo "developer ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/developer \
    && chmod 0440 /etc/sudoers.d/developer

# Switch to the non-root user
USER developer
WORKDIR /home/developer/workspace