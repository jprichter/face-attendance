#!/bin/bash

# This script sets the password for the 'postgres' user to match the default config.
# Run this with sudo.

DB_PASS="password"

echo "Setting password for user 'postgres' to '$DB_PASS'..."

sudo -u postgres psql -c "ALTER USER postgres PASSWORD '$DB_PASS';"

echo "Password updated successfully."
echo "If you still get authentication errors, check your /etc/postgresql/*/main/pg_hba.conf"
echo "to ensure 'md5' or 'scram-sha-256' is allowed for 127.0.0.1."
