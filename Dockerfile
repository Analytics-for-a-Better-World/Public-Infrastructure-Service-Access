############################################
# BUILDER & RUNTIME (Single-stage build with Poetry)
############################################

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for geospatial packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libgdal-dev \
    gdal-bin \
    libspatialindex-dev \
    libproj-dev \
    libgeos-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
RUN export GDAL_VERSION=$(gdal-config --version) && \
    echo "GDAL_VERSION=${GDAL_VERSION}" >> /etc/environment
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Configure Poetry to not create a virtual environment inside Docker
RUN poetry config virtualenvs.create false

# Copy the entire project first to ensure README.md and other files are available
COPY . ./

# Install dependencies with Poetry
RUN poetry install --no-interaction --no-ansi

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

WORKDIR /app/pisa_app

ENTRYPOINT ["streamlit", "run", "main_page.py", "--server.port=8501"]
