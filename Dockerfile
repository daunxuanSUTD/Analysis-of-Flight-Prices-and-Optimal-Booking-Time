# Use a lightweight Python 3.11 base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Environment variables to make Python behave nicely in containers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for building some Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency file and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and required resources into the container
COPY api.py .
COPY models ./models
COPY templates ./templates

# (Optional) If you need any other folders, copy them here as well, e.g.:
# COPY output ./output

# Expose the port used by Uvicorn inside the container
EXPOSE 8000

# Start the FastAPI app with Uvicorn when the container runs
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
