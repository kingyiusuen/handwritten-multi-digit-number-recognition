# Base image
FROM python:3.8-slim

# Install gcc (required for editdistance)
RUN apt-get update
RUN apt-get install -y --no-install-recommends build-essential

# Install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install fastapi==0.66.0 uvicorn==0.14.0

# Copy only the relevant directories
COPY handwritten_digit_string_recognition ./handwritten_digit_string_recognition
COPY server_api/ ./
COPY artifacts ./artifacts

# Export ports
EXPOSE 5000

ENTRYPOINT ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "5000"]