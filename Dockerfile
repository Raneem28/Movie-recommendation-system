# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860

# Create the user with UID 1000 (Required for Hugging Face Spaces)
RUN useradd -m -u 1000 user

# Set working directory to the user's home
WORKDIR /home/user/app

# Copy requirements file
COPY requirements-docker.txt requirements.txt

# Install python dependencies as root (easiest way) but accessible to user
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the rest of the application code with permission
# We copy specific folders to avoid copying venv or tmp files if .dockerignore fails
COPY --chown=user api/ ./api/
COPY --chown=user models/ ./models/
COPY --chown=user ml-32m-split/ ./ml-32m-split/
COPY --chown=user data_enrichment/ ./data_enrichment/
COPY --chown=user templates/ ./templates/
COPY --chown=user static/ ./static/
COPY --chown=user tests/ ./tests/

# Switch to the non-root user
USER user

# Expose the port (7860 is standard for HF Spaces)
EXPOSE 7860

# Command to run the application (Bind to 0.0.0.0 is crucial)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
