FROM python:3.9-slim

WORKDIR /app

# Fix the path for requirements.txt
COPY crop-prediction-api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the contents of the project directory
COPY crop-prediction-api/crop-prediction-api/ .

# Create necessary directories
RUN mkdir -p model data

EXPOSE 8000

# Update the command to reference the correct path
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
