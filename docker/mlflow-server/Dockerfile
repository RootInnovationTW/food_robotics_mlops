FROM python:3.9-slim

WORKDIR /app

COPY requirements-mlflow.txt .
RUN pip install -r requirements-mlflow.txt

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
