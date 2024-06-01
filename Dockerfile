# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    mdbtools \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8080 available to the world outside this container
EXPOSE 8080

# Define environment variable
ENV NAME MonteCarloSimulering

# Run the application using Gunicorn
CMD ["gunicorn", "-w", "4", "-t", "300", "-k", "gevent", "-b", "0.0.0.0:8000", "app:server"]
#CMD ["gunicorn", "-b", "0.0.0.0:8080", "app:server"]
#CMD ["gunicorn", "--workers=3", "--timeout=300", "app:server"]