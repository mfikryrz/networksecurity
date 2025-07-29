
# 🛡️ NetworkSecurity - End-to-End MLOps for Phishing Detection

This project implements a complete **MLOps pipeline** for detecting phishing websites using structured URL-based features. It includes data ingestion, preprocessing, model training, experiment tracking, containerization, deployment to AWS (ECR + EC2), and a REST API powered by FastAPI.

---

## 🚀 Project Overview

The goal is to build a production-grade system that:

* Ingests data from a MongoDB collection
* Validates and transforms the data
* Trains multiple classification models
* Tracks experiments using MLflow via [DagsHub](https://dagshub.com/)
* Builds a Docker image and deploys it using GitHub Actions to **Amazon ECR + EC2**
* Exposes REST API endpoints for:

  * `/train` — runs the full pipeline
  * `/predict` — receives a CSV file and returns predictions in a rendered HTML table

---

## 🔧 Tech Stack

| Layer                | Tools Used                                      |
| -------------------- | ----------------------------------------------- |
| **Data Source**      | MongoDB                                         |
| **Pipeline**         | Python, Scikit-learn, Pandas, NumPy             |
| **MLOps**            | GitHub Actions, AWS S3, AWS ECR, Docker, MLflow |
| **Experimentation**  | MLflow + DagsHub                                |
| **Web API**          | FastAPI, Jinja2                                 |
| **Containerization** | Docker                                          |
| **CI/CD**            | GitHub Actions → ECR → EC2 (Self-hosted runner) |

---

## 🧱 Pipeline Architecture

```mermaid
graph TD
  A[MongoDB Dataset] --> B[Data Ingestion]
  B --> C[Data Validation]
  C --> D[Data Transformation]
  D --> E[Model Training]
  E --> F[MLflow Tracking via DagsHub]
  E --> G[Model Saving]
  G --> H[S3 Sync]
  G --> I[Final Model (Pickle)]

  subgraph CI/CD [CI/CD Pipeline]
    J[GitHub Actions] --> K[Build Docker Image]
    K --> L[Push to ECR]
    L --> M[Deploy to EC2 Instance]
  end

  subgraph API [FastAPI Service]
    N["/train"] --> B
    O["/predict"] --> P[Load Model & Preprocessor] --> Q[Return Prediction as Table]
  end

  H --> R[S3 Bucket]
```

---

## 📁 Project Structure

```
networksecurity/
│
├── .github/workflows/main.yml       # GitHub Actions for CI/CD
├── app.py                           # FastAPI application
├── Dockerfile                       # Docker configuration
├── requirements.txt                 # List of dependencies (for local use)
├── networksecurity/                 # Main package
│   ├── components/                  # Data pipeline stages
│   ├── constant/                    # Global constants
│   ├── entity/                      # Config and artifact entities
│   ├── exception/                   # Custom exception class
│   ├── logging/                     # Logging utility
│   ├── pipeline/                    # Pipeline orchestration
│   ├── cloud/                       # AWS S3 syncer
│   └── utils/                       # Utility functions
├── templates/table.html             # HTML response template for predictions
└── data_schema/schema.yaml          # Schema config for validation
```

---

## ✅ Features

* [x] Data ingestion from MongoDB (with schema-based validation)
* [x] Drift detection using Kolmogorov-Smirnov test
* [x] KNN-based imputation via Scikit-learn pipeline
* [x] Automated model selection using GridSearchCV
* [x] MLflow experiment tracking integrated with DagsHub
* [x] End-to-end model training and deployment via `/train`
* [x] Real-time batch prediction via `/predict`

---

## 📜 License

This project is licensed under the MIT License.

---

Let me know if you'd like:

* Auto-generated OpenAPI schema
* Screenshots of API or training logs
* Badges (e.g., Docker, GitHub CI, Python version, DagsHub link) in header
