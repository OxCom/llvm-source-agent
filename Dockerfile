FROM python:3.11-slim AS builder

COPY ./ ./

RUN mkdir -p /app/wheels \
    && pip wheel --no-cache-dir --no-deps --wheel-dir /app/wheels -r requirements.txt


FROM python:3.11-slim

RUN mkdir -p /root/.cache/pip \
    && mkdir -p /app/code \
    && mkdir -p /app/source \
    && mkdir -p /app/index \
    && mkdir -p /app/wheels

RUN apt-get update \
    && apt-get install -y \
      git \
      curl  \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app/

COPY --from=builder /app/wheels /app/wheels
COPY requirements.txt ./

# Install dependencies with cache enabled
RUN pip install --no-cache /app/wheels/*

COPY src/ /app/code

CMD ["streamlit", "run", "/app/code/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
