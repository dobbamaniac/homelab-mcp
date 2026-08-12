FROM python:3.13-alpine3.22

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PEEK_ROOT=/workspace

RUN addgroup -S -g 10001 peek \
    && adduser -S -D -H -u 10001 -G peek peek

WORKDIR /app

COPY requirements.txt ./
RUN pip install \
    --no-cache-dir \
    --disable-pip-version-check \
    -r requirements.txt

COPY --chown=10001:10001 server.py ./

USER 10001:10001

EXPOSE 8000

CMD ["python", "/app/server.py"]
