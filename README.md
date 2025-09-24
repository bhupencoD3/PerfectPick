# Flipkart Product Recommendation System

## Overview

This repository contains an end-to-end **Flipkart Product Recommendation System**, designed to demonstrate the integration of modern AI frameworks, databases, monitoring, and container orchestration. The project showcases how to build, deploy, and monitor a scalable recommendation engine using a professional MLOps stack.

## Objectives

* Implement a recommendation pipeline for e-commerce products.
* Integrate **LangChain**, **Groq**, and **HuggingFace** for AI-powered reasoning and NLP capabilities.
* Use **AstraDB** for scalable, cloud-native NoSQL storage.
* Containerize the application with **Docker** and orchestrate with **Minikube/Kubernetes**.
* Deploy a simple **Flask + HTML/CSS** web interface.
* Monitor metrics using **Prometheus** and visualize them in **Grafana**.

## Tech Stack

* **AI/ML**: LangChain, HuggingFace, Groq
* **Database**: AstraDB (Cassandra NoSQL)
* **Backend**: Flask
* **Frontend**: HTML/CSS
* **Containerization & Orchestration**: Docker, Minikube, Kubectl
* **Monitoring & Visualization**: Prometheus, Grafana
* **Version Control**: GitHub

## Project Structure

```
flipkart-product-recommendation/
├── app/                  # Flask application code
│   ├── templates/        # HTML templates
│   ├── static/           # CSS/JS/static files
│   └── main.py           # Flask entry point
├── models/               # Recommendation/LLM integration modules
├── configs/              # Configuration files (dotenv, yaml)
├── notebooks/            # Experimental Jupyter notebooks
├── requirements.txt      # Python dependencies
├── setup.py              # Packaging configuration
├── environment.yml       # Conda environment (optional)
├── README.md             # Project documentation
└── MANIFEST.in           # Include non-Python files in package
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/bhupencoD3/flipkart-product-recommendation.git
cd flipkart-product-recommendation
```

### 2. Setup environment

Using Conda:

```bash
conda env create -f environment.yml -p ./venv
conda activate ./venv
```

Or using pip:

```bash
pip install -r requirements.txt
```

### 3. Run Flask app

```bash
python app/main.py
```

### 4. Access the application

Open your browser and navigate to:

```
http://127.0.0.1:5000/
```

## Monitoring

* Prometheus collects metrics from the app.
* Grafana dashboards provide visualization of system and model performance.

## Contribution

Contributions are welcome! Please open issues or submit PRs with improvements.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
