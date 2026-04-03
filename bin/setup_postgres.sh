#!/bin/bash

# Exit on error
set -e

DB_NAME="face_attendance"
PGVECTOR_VERSION="v0.5.1"

echo "Checking operating system..."
OS_TYPE=$(uname -s)

if [[ "$OS_TYPE" == "Linux" ]]; then
    # Assume Ubuntu/Debian
    echo "Detected Ubuntu Linux."
    sudo apt update
    sudo apt install -y postgresql postgresql-contrib postgresql-server-dev-all gcc make git
    
    # Install pgvector from source
    echo "Installing pgvector from source..."
    cd /tmp
    rm -rf pgvector
    git clone --branch $PGVECTOR_VERSION https://github.com/pgvector/pgvector.git
    cd pgvector
    make
    sudo make install
    cd -
    
    # Start Postgres
    sudo systemctl start postgresql
    sudo systemctl enable postgresql

elif [[ "$OS_TYPE" == "Darwin" ]]; then
    # macOS
    echo "Detected macOS."
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Please install it first: https://brew.sh/"
        exit 1
    fi
    brew install postgresql
    brew install pgvector
    
    # Start Postgres
    brew services start postgresql

else
    echo "Unsupported OS: $OS_TYPE"
    exit 1
fi

# Create Database and Extensions
echo "Setting up database '$DB_NAME'..."

# Try to create database (ignore error if it exists)
sudo -u postgres psql -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1 || sudo -u postgres psql -c "CREATE DATABASE $DB_NAME;"

# Run schema and extension (from docs/database.md)
sudo -u postgres psql -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Create tables
echo "Creating tables..."
sudo -u postgres psql -d $DB_NAME <<EOF
CREATE TABLE IF NOT EXISTS members (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    face_embedding vector(512),
    enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    image_path TEXT
);

CREATE TABLE IF NOT EXISTS attendance_log (
    id SERIAL PRIMARY KEY,
    member_id INTEGER REFERENCES members(id) ON DELETE CASCADE,
    check_in_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS unknown_detections (
    id SERIAL PRIMARY KEY,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    face_embedding vector(512),
    image_path TEXT,
    group_id UUID
);
EOF

echo "PostgreSQL setup complete!"
