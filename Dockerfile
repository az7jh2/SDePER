FROM python:3.9.12-slim-bullseye
RUN mkdir /data && chmod -R 755 /data && pip install --no-cache-dir cvae-glrm