# Airflow Pipeline Setup and Usage

This document provides instructions for setting up, running, and troubleshooting the Airflow-based data pipeline for the **House Price Prediction MLOps** project. Apache Airflow orchestrates various ETL processes and machine learning workflows, ensuring modular, scalable, and efficient data processing.

---

## 1. Introduction

The Airflow pipeline automates the end-to-end data processing and machine learning workflow for the House Price Prediction project. Key stages in the pipeline include:
- **Data acquisition** from Google Cloud Storage (GCS)
- **Data preprocessing** (cleaning, validation, splitting)
- **Feature engineering** and **data augmentation**
- **Model training** and **evaluation**
- **Artifact storage** and version control using DVC

---

## 2. Pipeline Overview

The pipeline consists of modular DAGs (Directed Acyclic Graphs) that run specific tasks in a sequence to process data, train models, and track outputs. Each DAG is designed to operate independently, allowing for flexibility in updating and testing individual components.

### DAGs Included:
- **data_prep_dag**: Handles data loading, validation, cleaning, and splitting.
- **feature_and_augm_dag**: Performs feature engineering and data augmentation.

### Pipeline Flow Diagram:
![Pipeline Diagram](https://github.com/user-attachments/assets/a8ecfda0-bf73-4ce4-9c8c-46b162cf10a2)
![Pipeline Diagram](https://github.com/user-attachments/assets/cf91bad9-c892-41cb-9640-31a8ef25fee9)

---

## 3. Setup

### a. Prerequisites

- **Python 3.8+**
- **Apache Airflow** (version 2.x)
- **Docker** (if using Docker Compose for containerized Airflow)
- **Google Cloud SDK** (for integration with GCS and DVC)

### b. Environment Variables

To facilitate connections to GCS and manage Airflow configuration, add environment variables in a `.env` file within the `config/` folder. Key variables include:
- `GCP_PROJECT_ID`
- `GCS_BUCKET_NAME`
- `DVC_REMOTE_PATH`
- Additional Google Cloud credentials as needed

### c. Installation Steps

1. **Clone the Repository**:
   ```bash
   git clone [REPO_URL]
   cd Airflow
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize Airflow**:
   - Set up the Airflow database, then start the scheduler and web server:
   ```bash
   airflow db init
   airflow scheduler &
   airflow webserver
   ```

4. **Docker Setup** (Optional):
   - If using Docker Compose, start Airflow services as follows:
   ```bash
   docker-compose up -d
   ```

5. **Configure DVC**:
   - Refer to the main project `README.md` for DVC setup instructions with Google Cloud.

---

## 4. Running the Pipeline

1. **Access the Airflow Web UI**:
   - Open `http://localhost:8080` in your browser to view your DAGs.

2. **Enable DAGs**:
   - In the Airflow UI, toggle the DAGs to enable them:
     - `data_prep_dag`: Manages data loading, cleaning, and splitting
     - `feature_and_augm_dag`: Conducts feature engineering and augmentation

3. **Trigger DAG Runs**:
   - Manually start a DAG run or configure a schedule in the DAG definitions as needed.

---

## 5. DAG Structure and Task Descriptions

### **data_prep_dag**
This DAG performs core data processing tasks:
- **Data Loading**: Loads raw data from GCS.
- **Data Validation**: Validates data schema and format.
- **Data Preprocessing**: Cleans missing values, encodes categorical data, and normalizes features.
- **Data Splitting**: Divides data into training, validation, and test sets.

### **feature_and_augm_dag**
This DAG is responsible for feature engineering and data augmentation:
- **Feature Engineering**: Applies transformations and extracts meaningful features.
- **Data Augmentation**: Expands the dataset using augmentation techniques to improve model robustness.

---

## 6. Troubleshooting and Common Issues

- **Airflow DAG Not Showing**: Ensure the `dags` folder is correctly set in `AIRFLOW_HOME`. Check for syntax errors in the DAG files.
- **GCS Permission Issues**: Verify that the GCP service account has access permissions to the specified GCS bucket.
- **Docker Errors**: If using Docker, confirm that the Docker daemon is running and `.env` variables are correctly configured.

---

## 7. Folder Structure (Airflow-Specific)

Here is an outline of the `Airflow` folder structure for easy reference:

```
Airflow
├── dags                   # Contains all DAG scripts
│   ├── __pycache__        # Python cache files
│   ├── data               # Data files required by DAGs
│   ├── src                # Source scripts for custom processing
│   │   ├── __init__.py
│   │   ├── data_augment.py
│   │   ├── data_prep.py
│   │   ├── data_splitting.py
│   │   ├── feature_select.py
│   │   └── label_encode.py
│   ├── __init__.py        # Initialization file for Airflow
│   ├── data_prep_dag.py   # DAG for data loading and preprocessing
│   └── feature_and_augm_dag.py  # DAG for feature engineering and augmentation
```

---

## 8. Testing and Validation

We use **GitHub Actions** to automate unit tests and validate each task within the DAGs. When changes are committed to the main branch, GitHub Actions triggers the tests and provides feedback on their results.

To run tests locally:
```bash
pytest tests/
```

![Unit Test Status](https://github.com/user-attachments/assets/74c31c68-e3ad-4c8e-ac46-17482e6718f7)

---

## 9. Resources and References

- **[Apache Airflow Documentation](https://airflow.apache.org/docs/)**: Official documentation for setting up and using Apache Airflow.
- **[Google Cloud SDK](https://cloud.google.com/sdk/docs)**: Guide for configuring GCS and managing data storage.
- **[DVC Documentation](https://dvc.org/doc)**: Guide for setting up DVC for data versioning.

# **Running MLflow UI in an Airflow Container**

This guide explains how to run the **MLflow UI** from an Airflow container in a Dockerized environment. It assumes you already have an Airflow setup running using `docker-compose` and MLflow installed in the containers via `_PIP_ADDITIONAL_REQUIREMENTS`.

---

## **Prerequisites**

1. **Airflow Setup**: Ensure your Airflow containers are running and properly configured in the `docker-compose.yaml` file.
2. **MLflow Installed**: MLflow should be included in the `_PIP_ADDITIONAL_REQUIREMENTS` of your Airflow setup:
   ```yaml
   _PIP_ADDITIONAL_REQUIREMENTS: "mlflow pandas scikit-learn kneed google-cloud-storage"
   ```
3. **Shared Volume**: The `mlruns` directory should be mounted to `/opt/airflow/mlruns` for consistent tracking:
   ```yaml
   volumes:
     - ./mlruns:/opt/airflow/mlruns
   ```

---

## **Steps to Run MLflow UI**

### **1. Start Airflow Containers**
Start your Airflow setup:
```bash
docker-compose up -d
```

Ensure all containers are running:
```bash
docker ps
```

### **2. Access the Desired Container**
Decide which Airflow container will host the MLflow UI. Typically, this will be the **worker** or **webserver** container. Enter the container:
```bash
docker exec -it airflow-airflow-worker-1 bash
```
*(Replace `airflow-airflow-worker-1` with the name of the desired container.)*

### **3. Start the MLflow UI**
Inside the container, start the MLflow UI:
```bash
mlflow ui --backend-store-uri postgresql+psycopg2://airflow:airflow@postgres/mlflow --host 0.0.0.0
```

- **`--backend-store-uri`**: Specifies the location of the MLflow tracking directory.
- **`--host`**: Ensures the UI is accessible from outside the container.

### **4. Access the MLflow UI**
The UI will be available at:
```
http://localhost:5000
```

If port **5000** is already in use, you can specify a different port:
```bash
mlflow ui --backend-store-uri file:///opt/airflow/mlruns --host 0.0.0.0 --port 5001
```
Ensure the port is mapped in your `docker-compose.yaml`:
```yaml
ports:
  - "5001:5001"
```

---

## **Stopping the MLflow UI**

### **1. If Running in the Foreground**
Stop the process by pressing:
```
Ctrl + C
```

### **2. If Running in the Background**
Find the MLflow process ID (PID) and terminate it:
```bash
ps aux | grep 'mlflow'
kill <PID>
```

---

## **Notes**

- For modular setups, consider running the MLflow UI in a dedicated container for better scalability and separation of concerns.
- Make sure the `mlruns` directory is shared between the container running the MLflow UI and the container where Airflow logs experiments.

---

## **Troubleshooting**

1. **Cannot Access MLflow UI**:
   - Verify that port **5000** is mapped correctly in the `docker-compose.yaml`.
   - Check if another process is using port **5000**:
     ```bash
     sudo ss -tuln | grep 5000
     ```

2. **Permission Issues**:
   Ensure the `mlruns` directory has the correct permissions:
   ```bash
   chmod -R 755 ./mlruns
   ```

---

# Troubleshooting `init-db.sh` Script in Docker Compose

This guide helps you troubleshoot issues related to the execution of the `init-db.sh` script in a Docker Compose setup, specifically for initializing a PostgreSQL database.

## Common Issues and Resolutions

### 1. File Permissions
Ensure the `init-db.sh` file has executable permissions.

**Command to check permissions:**  
```bash
ls -l init-db.sh
```

Expected output:  
```bash
-rwxr-xr-x 1 user group size date init-db.sh
```

If permissions are incorrect, run:  
```bash
chmod +x init-db.sh
```

### 2. Line Endings
Ensure the script has Unix line endings. Windows-style line endings can cause syntax errors in the container.

**Convert to Unix format:**  
Install `dos2unix` (if not already installed):  
```bash
sudo apt install dos2unix
```

Run the conversion:  
```bash
dos2unix init-db.sh
```

### 3. Correct Placement in Docker Compose
Ensure the script is mounted correctly in the `docker-compose.yml` file.

Example:  
```yaml
services:
  postgres:
    volumes:
      - ./scripts/init-db.sh:/docker-entrypoint-initdb.d/init-db.sh
```

### 4. Script Debugging
If the script fails to execute, check the logs of the PostgreSQL container:  
```bash
docker logs <postgres-container-id>
```

Manually execute the script inside the container to debug errors:  
```bash
docker exec -it <postgres-container-id> bash
bash /docker-entrypoint-initdb.d/init-db.sh
```

### 5. Confirm Database Initialization
Check if the `mlflow` database was successfully created:  
```bash
docker exec -it <postgres-container-id> psql -U <username>
\l
```

The output should list `mlflow` as one of the databases.

## Sample `init-db.sh` Script
```bash
#!/bin/bash
# Check if the database has already been initialized
if [ ! -f "/var/lib/postgresql/data/.db_initialized" ]; then
  echo "Initializing database..."
  psql -U airflow -c "CREATE DATABASE mlflow;"
  psql -U airflow -c "GRANT ALL PRIVILEGES ON DATABASE mlflow TO airflow;"
  touch /var/lib/postgresql/data/.db_initialized
else
  echo "Database already initialized."
fi
```

## Notes
- Ensure the `init-db.sh` script is idempotent to avoid issues during multiple runs.
- The `.db_initialized` file acts as a flag to ensure the database initialization runs only once.

## Contact
If you encounter further issues, consult the Docker and PostgreSQL documentation or reach out for support.


