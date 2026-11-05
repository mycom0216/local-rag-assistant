FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Ollama 는 호스트에서 실행 (GPU 사용). 컨테이너에서는 host.docker.internal 로 접근
ENV OLLAMA_URL=http://host.docker.internal:11434
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
