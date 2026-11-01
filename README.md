<<<<<<< HEAD
# CloudSentinel AI — Network Traffic Analytics Mini Project

A beginner-friendly college prototype combining **Data Mining, Data Visualization, and Cloud Computing/DevOps**.

## What this project does
- Loads a CSV file of network traffic records.
- Trains a Random Forest classifier using a labeled dataset.
- Evaluates it with a train/test split and reports precision, recall, F1 and a confusion matrix.
- Provides a Streamlit dashboard for CSV upload, predictions, and visualizations.
- Includes a FastAPI prediction API, Dockerfiles, Docker Compose, tests, and a GitHub Actions CI workflow.
- Includes optional AWS deployment guidance.

**Important limitations:** This is an educational offline dataset-analysis prototype, not a production intrusion-prevention system. It does not monitor live networks or block attacks. Predictions depend on dataset quality and feature compatibility. Only use datasets you are permitted to use.

## Quick start (Windows 11 / CMD)

### 1. Install Python
Install Python 3.10 or 3.11 and select **Add Python to PATH** during installation.

### 2. Extract the ZIP
Extract `CloudSentinel-AI.zip` to a folder, then open CMD in that folder.

### 3. Create a virtual environment
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Add a dataset
Download the UCI RT-IoT2022 dataset from:
https://archive.ics.uci.edu/dataset/942/rt-iot2022

Put the CSV file in `data/` and rename it `network_traffic.csv`.

The dataset needs a label column. The project tries common names such as `Attack_type`, `label`, `Label`, `class`, or `target`. If the actual label column uses a different name, edit `TARGET_COLUMN` in `src/config.py`.

### 5. Train and evaluate
```bat
python -m src.train_model --data data/network_traffic.csv
```

The model pipeline and metadata are saved to `artifacts/`.

### 6. Run the dashboard
```bat
streamlit run dashboard/app.py
```
Open the local URL shown by Streamlit (usually http://localhost:8501).

### 7. Run the API (optional)
Open a second CMD window, activate `.venv`, then:
```bat
uvicorn api.main:app --reload
```
API docs: http://127.0.0.1:8000/docs

The API requires a trained model in `artifacts/`. Upload one CSV file using `/predict-csv`.

## Use your own CSV
- Training CSV: should contain multiple feature columns and a target/label column.
- Prediction CSV: should contain the same input feature columns used during training, without requiring the label column.
- Do not include personal or confidential network information.
- This prototype intentionally limits uploaded file size and validates file type.

## Docker (optional)
Docker Desktop must be installed and running.

Run dashboard + API:
```bat
docker compose up --build
```
Dashboard: http://localhost:8501
API docs: http://localhost:8000/docs

Train the model locally first so `artifacts/model.joblib` exists. The `artifacts/` folder is mounted into the containers.

## AWS deployment overview
A typical student deployment:
1. Build and test the app locally.
2. Create an ECR repository and push the Docker image.
3. Create an ECS Fargate service with suitable task networking.
4. Configure a security group to allow only the required web traffic.
5. Store dataset/model artifacts in a private S3 bucket if needed.
6. Use IAM task roles rather than embedding AWS keys in code.
7. Send application logs to CloudWatch.
8. Configure GitHub Actions deployment only after manual deployment works.

AWS resources may incur charges. Set a budget alert, use least-privilege IAM, and delete resources after the demonstration. A budget alert is not a spending cap.

## Repository layout
```text
CloudSentinel-AI/
├── api/main.py
├── dashboard/app.py
├── src/config.py
├── src/data_utils.py
├── src/train_model.py
├── src/predict.py
├── tests/
├── data/README.md
├── artifacts/README.md
├── Dockerfile
├── Dockerfile.api
├── docker-compose.yml
├── requirements.txt
└── .github/workflows/ci.yml
```

## Before submitting
- Replace the demo data with the approved dataset.
- Record actual evaluation metrics; never invent results.
- Explain false positives, false negatives, dataset limitations, and why a suspicious prediction is not proof of an attack.
- Add your names, UIDs, university, screenshots, and report only after the application runs successfully.
=======
# CloudSentinel-AI
AI-powered Network Intrusion Detection and Traffic Classification System
>>>>>>> c78dcb9338ad54ce3d0173dba91bfa60b609b9b7
