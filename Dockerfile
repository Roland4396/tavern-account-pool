FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py channel_config.py channel_catalog.py provider_capacity.py provider_wire.py model_rules.py ./
COPY pools.example.json ./pools.json
EXPOSE 3200
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3200"]
