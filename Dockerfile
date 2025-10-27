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
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to PATH
ENV PATH="/root/.local/bin:$PATH"

# Configure Poetry
RUN poetry config virtualenvs.create false

# Copy the entire project and install dependencies
COPY . ./
RUN poetry install --no-interaction --no-ansi

# Fix for streamlit-folium missing marker images
RUN mkdir -p /usr/local/lib/python3.10/site-packages/streamlit_folium/frontend/build && \
    curl -o /usr/local/lib/python3.10/site-packages/streamlit_folium/frontend/build/marker-icon-2x.png https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png && \
    curl -o /usr/local/lib/python3.10/site-packages/streamlit_folium/frontend/build/marker-shadow.png https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png && \
    curl -o /usr/local/lib/python3.10/site-packages/streamlit_folium/frontend/build/marker-icon.png https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

WORKDIR /app/pisa_app

ENTRYPOINT ["streamlit", "run", "main_page.py", "--server.port=8501"]
