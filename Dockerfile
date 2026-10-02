FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    APP_PORT=8080

WORKDIR /app

# 应用仅依赖 Python 标准库，无需 pip install。
COPY app ./app
COPY tests ./tests
COPY verify.py ./verify.py

EXPOSE 8080

# 容器内健康检查：供 docker compose 的 service_healthy 条件依赖
HEALTHCHECK --interval=3s --timeout=3s --start-period=3s --retries=20 \
  CMD python -c "import json,os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('APP_PORT','8080')+'/health',timeout=2).read()" || exit 1

CMD ["python", "-m", "app.server"]
