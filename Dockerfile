# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install system dependencies that might be needed by Python packages
RUN apt-get update && apt-get install -y --no-install-recommends     gcc     libpoppler-cpp-dev     pkg-config     tesseract-ocr  && apt-get clean  && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port the app runs on (if any, for now not specified but good practice)
# EXPOSE 8000

# Specify the command to run on container startup
# This will depend on how main.py is typically invoked.
# Assuming it's a CLI application, we might not need a default CMD
# or we can set it to run a basic command like "python main.py --help"
CMD ["python", "main.py", "--help"]
