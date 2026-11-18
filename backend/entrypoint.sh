#!/bin/sh
set -e

if [ "${RUN_MIGRATIONS}" = "1" ]; then
    echo "Running database migrations..."
    alembic upgrade head
else
    echo "Skipping migrations (RUN_MIGRATIONS not set to 1)"
fi

echo "Starting application..."
exec "$@"