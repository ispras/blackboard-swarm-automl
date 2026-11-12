# syntax=docker/dockerfile:1.7
#
# One Dockerfile, seven targets:
#   base       opencode + Python + repo scripts (no ML stack)
#   ml         base + the full MLE-bench Python stack
#   min        base + a light Python stack (offline toy demo)
#   swarm      ml + systemd + supervisor   (main swarm container, PID 1 = systemd)
#   tools      ml + mle-bench checkout     (prepare/grade)
#   swarm-min  min + systemd               (toy swarm container)
#   tools-min  min + mle-bench checkout
#
# `ml` and `tools` are intentionally separate images: the swarm container must
# NOT contain the private data or the mle-bench cache. The tools/grader
# containers get both through bind mounts.

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-bookworm AS base

ARG SWARM_UID=1000
ARG SWARM_GID=1000
ARG NODE_VERSION=20.17.0
ARG OPENCODE_VERSION=1.18.34

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl xz-utils git sqlite3 procps tini \
    && rm -rf /var/lib/apt/lists/*

# Node + opencode CLI. The harness starts `opencode serve` and attaches agents
# to it over localhost inside the container.
RUN curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" -o /tmp/node.tar.xz \
    && tar -xJf /tmp/node.tar.xz -C /usr/local --strip-components=1 \
    && rm -f /tmp/node.tar.xz \
    && npm install -g --unsafe-perm "opencode-ai@${OPENCODE_VERSION}" \
    && npm cache clean --force \
    && node --version \
    && npm --version \
    && opencode --version

WORKDIR /workspace
COPY bin /workspace/bin
COPY prompts /workspace/prompts
RUN find /workspace/bin -type f -name '*.sh' -exec chmod +x {} + || true

# The container runs as the host UID/GID so bind-mounted files stay writable.
RUN (getent group "${SWARM_GID}" >/dev/null || groupadd -g "${SWARM_GID}" agent) \
    && (getent passwd "${SWARM_UID}" >/dev/null \
        || useradd -M -u "${SWARM_UID}" -g "${SWARM_GID}" -d /home/agent -s /bin/bash agent) \
    && (getent group video >/dev/null && usermod -aG video agent || true) \
    && mkdir -p /home/agent/.cache /home/agent/.local/share /home/agent/.config \
    && chown -R "${SWARM_UID}:${SWARM_GID}" /home/agent

ENV HOME=/home/agent \
    XDG_DATA_HOME=/home/agent/.local/share \
    XDG_CACHE_HOME=/home/agent/.cache

USER ${SWARM_UID}:${SWARM_GID}
CMD ["sleep", "infinity"]


# --------------------------------------------------------------------------- full ML stack
FROM base AS ml

ARG SWARM_UID=1000
ARG SWARM_GID=1000
USER root

# Runtime libraries used by the MLE-bench stack: opencv, poppler/pdf2image,
# tesseract/pyocr, librosa/resampy/torchaudio.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libsm6 libxext6 poppler-utils tesseract-ocr \
      libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY docker/requirements-full.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip wheel \
    && python3 -m pip install -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt

USER ${SWARM_UID}:${SWARM_GID}
CMD ["sleep", "infinity"]


# --------------------------------------------------------------------------- swarm runtime (systemd)
FROM ml AS swarm

ARG SWARM_UID=1000
ARG SWARM_GID=1000
ARG PYENV_VERSION=v2.8.15
USER root

# The main swarm container runs systemd as PID 1, so `systemd-run --user
# --scope` is the real command. At run time this needs a privileged container
# with cgroup access and tmpfs /run (see compose.yaml).
RUN apt-get update && apt-get install -y --no-install-recommends \
      systemd systemd-sysv dbus dbus-user-session libpam-systemd \
    && rm -rf /var/lib/apt/lists/*

# Real pyenv with a `sis-black` environment. The environment is a venv with
# system-site-packages, so it reuses the ML stack installed above.
RUN git clone --depth 1 --branch "${PYENV_VERSION}" https://github.com/pyenv/pyenv.git /home/agent/.pyenv \
    && rm -rf /home/agent/.pyenv/.git \
    && chown -R "${SWARM_UID}:${SWARM_GID}" /home/agent/.pyenv

USER ${SWARM_UID}:${SWARM_GID}
RUN python3 -m venv --system-site-packages /home/agent/.pyenv/versions/sis-black

USER root
COPY docker/systemd/pyenv.sh /etc/profile.d/pyenv.sh
RUN chmod 0644 /etc/profile.d/pyenv.sh

COPY docker/systemd/swarm.service.in /tmp/swarm.service.in
COPY docker/systemd/runtime-dir.conf /tmp/runtime-dir.conf
RUN sed "s/__SWARM_UID__/${SWARM_UID}/g" /tmp/swarm.service.in > /etc/systemd/system/swarm.service \
    && mkdir -p "/etc/systemd/system/user@${SWARM_UID}.service.d" \
    && sed "s/__SWARM_UID__/${SWARM_UID}/g" /tmp/runtime-dir.conf \
       > "/etc/systemd/system/user@${SWARM_UID}.service.d/runtime-dir.conf" \
    && mkdir -p /etc/systemd/system/multi-user.target.wants /var/lib/systemd/linger \
    && ln -sf /etc/systemd/system/swarm.service /etc/systemd/system/multi-user.target.wants/swarm.service \
    && touch /var/lib/systemd/linger/agent

STOPSIGNAL SIGRTMIN+3
CMD ["/sbin/init"]


# --------------------------------------------------------------------------- tools / grader
FROM ml AS tools

ARG SWARM_UID=1000
ARG SWARM_GID=1000
ARG MLEBENCH_REPO=https://github.com/openai/mle-bench.git
ARG MLEBENCH_REF=507f92e1138bb6e40dac5c6ee7a6758e6424bf97
USER root

# mle-bench keeps leaderboards in Git LFS; without `git lfs pull` the
# leaderboard.csv files are pointer files and ranking/grading fails.
RUN apt-get update && apt-get install -y --no-install-recommends git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system \
    && git clone "${MLEBENCH_REPO}" /opt/mle-bench \
    && git -C /opt/mle-bench checkout --detach "${MLEBENCH_REF}" \
    && git -C /opt/mle-bench lfs pull \
    && rm -rf /opt/mle-bench/.git

ENV MLEBENCH_REPO=/opt/mle-bench \
    PYTHONPATH=/opt/mle-bench

USER ${SWARM_UID}:${SWARM_GID}
CMD ["sleep", "infinity"]


# --------------------------------------------------------------------------- light targets
FROM base AS min

ARG SWARM_UID=1000
ARG SWARM_GID=1000
USER root
COPY docker/requirements-min.txt /tmp/requirements-min.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --upgrade pip wheel \
    && python3 -m pip install -r /tmp/requirements-min.txt \
    && rm -f /tmp/requirements-min.txt
USER ${SWARM_UID}:${SWARM_GID}
CMD ["sleep", "infinity"]


# --------------------------------------------------------------------------- swarm-min (systemd)
FROM min AS swarm-min

ARG SWARM_UID=1000
ARG SWARM_GID=1000
ARG PYENV_VERSION=v2.8.15
USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
      systemd systemd-sysv dbus dbus-user-session libpam-systemd \
    && rm -rf /var/lib/apt/lists/*

# Real pyenv with a `sis-black` environment. The environment is a venv with
# system-site-packages, so it reuses the ML stack installed above.
RUN git clone --depth 1 --branch "${PYENV_VERSION}" https://github.com/pyenv/pyenv.git /home/agent/.pyenv \
    && rm -rf /home/agent/.pyenv/.git \
    && chown -R "${SWARM_UID}:${SWARM_GID}" /home/agent/.pyenv

USER ${SWARM_UID}:${SWARM_GID}
RUN python3 -m venv --system-site-packages /home/agent/.pyenv/versions/sis-black

USER root
COPY docker/systemd/pyenv.sh /etc/profile.d/pyenv.sh
RUN chmod 0644 /etc/profile.d/pyenv.sh

COPY docker/systemd/swarm.service.in /tmp/swarm.service.in
COPY docker/systemd/runtime-dir.conf /tmp/runtime-dir.conf
RUN sed "s/__SWARM_UID__/${SWARM_UID}/g" /tmp/swarm.service.in > /etc/systemd/system/swarm.service \
    && mkdir -p "/etc/systemd/system/user@${SWARM_UID}.service.d" \
    && sed "s/__SWARM_UID__/${SWARM_UID}/g" /tmp/runtime-dir.conf \
       > "/etc/systemd/system/user@${SWARM_UID}.service.d/runtime-dir.conf" \
    && mkdir -p /etc/systemd/system/multi-user.target.wants /var/lib/systemd/linger \
    && ln -sf /etc/systemd/system/swarm.service /etc/systemd/system/multi-user.target.wants/swarm.service \
    && touch /var/lib/systemd/linger/agent

STOPSIGNAL SIGRTMIN+3
CMD ["/sbin/init"]


FROM min AS tools-min

ARG SWARM_UID=1000
ARG SWARM_GID=1000
ARG MLEBENCH_REPO=https://github.com/openai/mle-bench.git
ARG MLEBENCH_REF=507f92e1138bb6e40dac5c6ee7a6758e6424bf97
USER root

# mle-bench keeps leaderboards in Git LFS; without `git lfs pull` the
# leaderboard.csv files are pointer files and ranking/grading fails.
RUN apt-get update && apt-get install -y --no-install-recommends git-lfs \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install --system \
    && git clone "${MLEBENCH_REPO}" /opt/mle-bench \
    && git -C /opt/mle-bench checkout --detach "${MLEBENCH_REF}" \
    && git -C /opt/mle-bench lfs pull \
    && rm -rf /opt/mle-bench/.git

ENV MLEBENCH_REPO=/opt/mle-bench \
    PYTHONPATH=/opt/mle-bench

USER ${SWARM_UID}:${SWARM_GID}
CMD ["sleep", "infinity"]
