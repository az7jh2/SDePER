# Incorporate a time delay in Docker Automated Builds to prevent the installation of outdated releases
FROM python:3.9.12-slim-bullseye
RUN sleep 300
RUN mkdir /data && chmod -R 755 /data && pip install --no-cache-dir sdeper