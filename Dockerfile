FROM python:3.10-slim

# Install system dependencies for librosa, soundfile, etc.
RUN apt-get update && \
    apt-get install -y libsndfile1 ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --root-user-action=ignore -r requirements.txt

# Expose the FastAPI port
EXPOSE 8000

# Start the server using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
