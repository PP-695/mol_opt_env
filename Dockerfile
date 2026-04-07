ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app/env
ARG BUILD_MODE=in-repo
ENV UV_LINK_MODE=copy

RUN if ! command -v uv >/dev/null 2>&1; then \
        curl -LsSf https://astral.sh/uv/install.sh | sh && \
        mv /root/.local/bin/uv /usr/local/bin/uv && \
        mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

COPY pyproject.toml uv.lock openenv.yaml README.md /app/env/
RUN mkdir -p /app/env/server

RUN if [ ! -f server/fpscores.pkl.gz ]; then \
        curl -fsSL -o server/fpscores.pkl.gz \
        "https://raw.githubusercontent.com/rdkit/rdkit/master/Contrib/SA_Score/fpscores.pkl.gz"; \
    fi

RUN uv venv /app/env/.venv

RUN if [ -f uv.lock ]; then \
        uv sync --frozen --no-install-project --no-editable; \
    else \
        uv sync --no-install-project --no-editable; \
    fi

COPY __init__.py client.py env.py inference.py models.py rubrics.py .env.example /app/env/
COPY server /app/env/server

RUN if [ -f uv.lock ]; then \
        uv sync --frozen --no-editable; \
    else \
        uv sync --no-editable; \
    fi

FROM ${BASE_IMAGE}

WORKDIR /app
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env
RUN rm -rf /app/env/.venv

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV PORT=8000
ENV HOST=0.0.0.0
ENV WORKERS=1
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["sh", "-c", "cd /app/env && python -m uvicorn server.app:app --host 0.0.0.0 --port 8000"]
