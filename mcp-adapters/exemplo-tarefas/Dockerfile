FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastmcp==4.1.5
COPY server.py .
EXPOSE 8765
CMD ["python", "server.py"]
