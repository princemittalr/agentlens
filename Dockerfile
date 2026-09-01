FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy app
COPY . .

# Expose port
EXPOSE 8000

# Start
CMD ["python3", "-m", "uvicorn", "dashboard.app:app", "--host", "0.0.0.0", "--port", "8000"]
