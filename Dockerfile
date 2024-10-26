# Use an official Python runtime as a base image
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .

# Copy the CSV file and the Python script into the container
COPY spam.csv .
COPY SpamDetection.py .
COPY templates ./templates

RUN pip install --no-cache-dir -r requirements.txt


# Expose the port on which the app will run
EXPOSE 8000

# Command to run the Flask app
CMD ["python", "SpamDetection.py"]
