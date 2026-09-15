# Nexus is a *worker*: one long-lived process that polls Telegram. It opens no
# HTTP port, so deploy it as a "worker" / "background service", never a web
# service -- a web service is health-checked on a port that will never open, and
# the platform will kill it in a restart loop.
FROM python:3.12-slim

# Send logs straight to the platform instead of sitting in a buffer, and skip
# writing .pyc files into the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first, so editing code does not invalidate the install layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Memory is a SQLite file. /data is where a host mounts a volume; without one
# the file still works, it just starts empty on every deploy.
ENV NEXUS_DB_PATH=/data/nexus.db
RUN useradd --create-home --uid 1000 nexus \
    && mkdir -p /data \
    && chown -R nexus:nexus /app /data \
    && chmod +x /app/entrypoint.sh \
    && command -v setpriv >/dev/null

# The bot runs as `nexus`, but the container *starts* as root: a volume
# mounted at /data arrives owned by root whatever the line above did, and only
# root can hand it over. entrypoint.sh does that, then drops to nexus.
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "bot.py"]
