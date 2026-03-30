FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ app/

# Expose port
EXPOSE 8000

# Run server — Railway sets $PORT automatically
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
