# Online Retail ML

An end-to-end data engineering and machine learning project based on the **Online Retail II** dataset. The project focuses on data cleaning, feature engineering, customer-level analysis, and **customer churn prediction**, with a FastAPI layer for serving analytical/model-related functionality.

## 📌 Project Overview

This project processes transactional e-commerce data and transforms it into structured datasets suitable for analysis and machine learning.

The main workflow is:

**Raw Data → Data Cleaning → Feature Engineering → Customer Features → Machine Learning → API**

The project is designed to demonstrate a practical machine learning pipeline, from raw transactional data to a trained predictive model.

## 🎯 Objectives

* Clean and prepare raw online retail transaction data.
* Handle missing, invalid, and inconsistent transaction records.
* Build customer-level features from transactional data.
* Perform exploratory data analysis.
* Train a machine learning model to predict customer churn.
* Store processed datasets and model metadata.
* Expose functionality through a REST API using FastAPI.
* Organize the project using a reproducible and maintainable structure.

## 🛠️ Technologies

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **FastAPI**
* **Pydantic**
* **Joblib**
* **Jupyter Notebook**
* **Parquet**
* **Docker**

## 📂 Project Structure

```text
online-retail-ml/
│
├── api/
│   ├── main.py
│   └── schemas.py
│
├── data/
│   ├── raw/
│   │   └── online_retail_II.xlsx
│   │
│   └── processed/
│       ├── customer_features.parquet
│       ├── returns_clean.parquet
│       ├── transactions_clean.parquet
│       └── etl_report.txt
│
├── models/
│   ├── churn_model.joblib
│   └── model_info.json
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── features/
│   │   └── build_features.py
│   │
│   └── models/
│       └── train.py
│
├── clean.py
├── load_data.py
├── Dockerfile
├── requirements.txt
└── .gitignore
```

> **Note:** Large datasets and generated model files are excluded from the Git repository using `.gitignore`.

## 🔄 Data Pipeline

### 1. Data Loading

The project starts from the **Online Retail II** transactional dataset.

The raw data contains information related to customer purchases, products, quantities, prices, dates, and transaction identifiers.

### 2. Data Cleaning

The cleaning stage prepares the transactional data for downstream analysis by handling data quality issues such as:

* Missing values
* Invalid transactions
* Returns/cancellations
* Incorrect quantities
* Inconsistent records

The cleaned transactions are stored in **Parquet** format for efficient processing.

### 3. Feature Engineering

Customer-level features are generated from the cleaned transactional data.

These features transform individual transactions into a customer-oriented dataset that can be used for machine learning.

Examples of customer behavior dimensions include:

* Purchase frequency
* Recency
* Monetary behavior
* Transaction activity
* Customer purchasing patterns

### 4. Exploratory Data Analysis

The Jupyter notebook provides exploratory analysis of the dataset and customer behavior.

The analysis helps identify patterns and relationships within the transactional data before applying machine learning.

### 5. Churn Prediction

A machine learning model is trained using the generated customer features.

The objective is to identify customers who may be at risk of becoming inactive based on their historical purchasing behavior.

The trained model is saved as:

```text
models/churn_model.joblib
```

Additional model information is stored in:

```text
models/model_info.json
```

## 🤖 Machine Learning Workflow

The machine learning workflow can be summarized as:

```text
Customer Transactions
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Customer-Level Dataset
        ↓
Train/Test Preparation
        ↓
Model Training
        ↓
Churn Prediction
        ↓
Saved Model
```

The training implementation is located in:

```text
src/models/train.py
```

Feature construction is handled in:

```text
src/features/build_features.py
```

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Dhiaakarray/online-retail-ml.git
cd online-retail-ml
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

### Data Processing

Run the data cleaning pipeline:

```bash
python clean.py
```

Build customer features:

```bash
python src/features/build_features.py
```

### Model Training

Train the churn prediction model:

```bash
python src/models/train.py
```

## 🔌 API

The project includes a **FastAPI** application located in:

```text
api/main.py
```

The API provides a layer for interacting with the project's analytical and machine-learning functionality.

To start the API locally:

```bash
uvicorn api.main:app --reload
```

Once running, the interactive API documentation can normally be accessed at:

```text
http://127.0.0.1:8000/docs
```

## 🐳 Docker

The project also includes a `Dockerfile` for containerized execution.

Build the Docker image:

```bash
docker build -t online-retail-ml .
```

Run the container:

```bash
docker run -p 8000:8000 online-retail-ml
```

## 📊 Dataset

The project is based on the **Online Retail II** dataset, a transactional e-commerce dataset containing historical retail purchases.

The dataset is used to study customer purchasing behavior and build customer-level features for churn prediction.

Because of repository size considerations, raw and processed datasets are excluded from version control.

## 🔒 Data & Model Files

The following types of files are intentionally excluded from GitHub:

```text
data/raw/
data/processed/
models/*.joblib
```

This keeps the repository lightweight while preserving the code required to reproduce the pipeline.

## 📈 Future Improvements

Potential improvements include:

* Adding more machine learning models for comparison.
* Hyperparameter optimization.
* Model evaluation and visualization.
* Feature importance analysis.
* Model performance monitoring.
* Containerized deployment.
* Cloud deployment.
* Automated ETL and model-training pipelines.
* CI/CD integration.

## 👤 Author

**Dhia Karray**

Computer Science student specializing in data-oriented technologies and machine learning.

GitHub:
https://github.com/Dhiaakarray

## 📄 License

This project is intended for educational and portfolio purposes.
