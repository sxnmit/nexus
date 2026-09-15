#!/bin/sh
# Container entrypoint: root for three lines, then `nexus` for everything else.
#
# A volume mounted at /data (Railway, `docker run -v`) arrives owned by root,
# whatever this image did to that directory at build time, so the bot's user
# could not create its database there ("unable to open database file"). Only
# root can hand the directory over, so the container starts as root, does
# exactly that, and drops privileges before the bot runs.
set -e

dir=$(dirname "${NEXUS_DB_PATH:-/data/nexus.db}")

if [ "$(id -u)" = "0" ]; then
    if [ "$dir" != "." ]; then
        mkdir -p "$dir"
        chown -R nexus:nexus "$dir"
    fi
    exec setpriv --reuid=nexus --regid=nexus --init-groups "$@"
fi

exec "$@"
