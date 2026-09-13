ARG PYTHON_VER="3.13"
ARG PYTHON_VER_UV="-p ${PYTHON_VER}t"

######## BUILDER OF PYTHON APP
FROM python:${PYTHON_VER}-slim AS builder

WORKDIR /opt


# Install system dependencies
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        gfortran \
        ccache \
        portaudio19-dev \
        pkg-config \
        libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/opt/.venv \
    CC="ccache gcc" \
    CXX="ccache g++" \
    FC="ccache gfortran"

#    UV_PYTHON_DOWNLOADS=never \

COPY pyproject.toml uv.lock ./

ARG PYTHON_VER_UV
ARG VERBOSE

RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    --mount=type=cache,id=ccache,target=/root/.cache/ccache \
    uv sync ${PYTHON_VER_UV:-} ${VERBOSE:-} --frozen --no-install-project --no-dev

RUN PYTHON_GIL=0 ${UV_PROJECT_ENVIRONMENT}/bin/python -c "\
import sys; \
import numpy; \
import scipy; \
import pyaudio; \
print(sys.version); \
print('GIL:', sys._is_gil_enabled()); \
print('numpy:', numpy.__version__); \
print('scipy:', scipy.__version__); \
print('pyaudio:', pyaudio.__version__) \
"

FROM python:${PYTHON_VER}-slim AS runner

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    procps \
    htop \
    && rm -rf /var/lib/apt/lists/*


#WORKDIR /app
#ARG _USER=appuser
#ARG _GROUP=appgroup
#RUN groupadd ${_GROUP} && useradd --no-log-init -r --no-create-home -g ${_GROUP} ${_USER} && \
#    mkdir ./data && \
#    chown -R  ${_USER}:${_GROUP} ./data


# Copy venv from previous stage "builder"
COPY --from=builder /opt/.venv /opt/.venv
COPY pyproject.toml .
COPY ./src src/
COPY --chmod=+x ./entrypoint.sh .

ENV PATH="/opt/.venv/bin:$PATH" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=./src PYTHON_GIL=0

#USER ${_USER}

CMD ["/app/entrypoint.sh"]

