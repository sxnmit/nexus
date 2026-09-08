# "worker", not "web": Nexus polls Telegram and serves no HTTP port.
# Used by buildpack/nixpacks deploys (Railway); Docker deploys use the Dockerfile
# CMD instead, and Render takes its start command from the dashboard.
worker: python bot.py
