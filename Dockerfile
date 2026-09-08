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

RUN useradd --create-home --uid 1000 nexus && chown -R nexus:nexus /app
USER nexus

CMD ["python", "bot.py"]
