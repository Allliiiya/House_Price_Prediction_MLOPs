#!/bin/bash
# Check if the database has already been initialized
if [ ! -f "/var/lib/postgresql/data/.db_initialized" ]; then
  echo "Initializing database..."

  # Create the 'mlflow' database if not already present
  psql -U airflow -c "CREATE DATABASE mlflow;"
  psql -U airflow -c "GRANT ALL PRIVILEGES ON DATABASE mlflow TO airflow;"

  # Mark the database as initialized by creating a file
  touch /var/lib/postgresql/data/.db_initialized

  echo "Database 'mlflow' created and initialized."
else
  # If the database is already initialized
  echo "Database 'mlflow' already initialized."
fi
