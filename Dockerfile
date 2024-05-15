############################################
# BUILDER
############################################

FROM python:3.10-slim AS builder

WORKDIR /app

# Install Poetry
RUN apt update && apt install -y curl
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Export dependencies as native `requirements.txt` file
COPY pyproject.toml poetry.lock ./
RUN poetry self add poetry-plugin-export
# Export without hashes to work around bug: https://github.com/python-poetry/poetry/issues/3472
RUN poetry export --without-hashes -f requirements.txt --output requirements.txt

############################################
# RUNTIME
############################################

FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY --from=builder /app/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy local code to the container image.
COPY . ./

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

WORKDIR /app/gpbp_app

ENTRYPOINT ["streamlit", "run", "main_page.py", "--server.port=8501"]
