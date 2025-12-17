# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy requirements file (Critical step!)
COPY requirements-docker.txt requirements.txt

# Install python dependencies
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the rest of the application code
# We copy specific folders to avoid copying venv or tmp files if .dockerignore fails
COPY api/ ./api/
COPY models/ ./models/

COPY data_enrichment/ ./data_enrichment/
COPY templates/ ./templates/
COPY static/ ./static/
COPY ml-32m-split/ ./ml-32m-split/

# Create a non-root user for security
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
