
-----

# PerfectPick: AI-Powered Phone Recommendation System

**Introduction**

PerfectPick is a production-grade AI recommendation system designed to deliver personalized smartphone recommendations from **Flipkart's** product catalog. It integrates classical information retrieval (**BM25**), modern semantic embeddings (**BGE**), and large language model (**LLM**)-based generation to provide accurate, context-aware recommendations. The system is built for scalability, leveraging a modular architecture deployed on **Google Cloud Platform (GCP)** using **Docker**, **Kubernetes**, **Prometheus**, and **Grafana**. The project is live at [http://34.14.203.32:8080/](http://34.14.203.32:8080/), accessible for testing and evaluation.

This README provides a comprehensive guide to PerfectPick’s architecture, components, deployment workflow, and monitoring setup, suitable for researchers, developers, and DevOps engineers. It covers the system from data ingestion to production, with a focus on reproducibility, scalability, and observability.

-----

## Table of Contents

  * [Project Overview](https://www.google.com/search?q=%23project-overview)
  * [Core Features](https://www.google.com/search?q=%23core-features)
  * [System Architecture](https://www.google.com/search?q=%23system-architecture)
  * [Directory Structure](https://www.google.com/search?q=%23directory-structure)
  * [Component-Level Breakdown](https://www.google.com/search?q=%23component-level-breakdown)
      * [Core Application Modules](https://www.google.com/search?q=%23core-application-modules)
      * [Utility Layer](https://www.google.com/search?q=%23utility-layer)
      * [API and Frontend](https://www.google.com/search?q=%23api-and-frontend)
      * [Data and Experiments](https://www.google.com/search?q=%23data-and-experiments)
      * [Testing Suite](https://www.google.com/search?q=%23testing-suite)
      * [Deployment Layer](https://www.google.com/search?q=%23deployment-layer)
  * [Technology Stack](https://www.google.com/search?q=%23technology-stack)
  * [Setup and Local Development](https://www.google.com/search?q=%23setup-and-local-development)
  * [Deployment Workflow](https://www.google.com/search?q=%23deployment-workflow)
  * [Monitoring and Observability](https://www.google.com/search?q=%23monitoring-and-observability)
  * [Scalability and Performance](https://www.google.com/search?q=%23scalability-and-performance)
  * [Future Improvements](https://www.google.com/search?q=%23future-improvements)
  * [License](https://www.google.com/search?q=%23license)

-----

## Project Overview

PerfectPick is an end-to-end recommendation system that processes user queries (e.g., "best budget phone under 15K with good camera") to suggest relevant smartphones from a dataset of **3903 Flipkart products (1456 unique models)**. It combines sparse retrieval (**BM25**), dense vector search (**Astra DB**), and **LLM-based generation** (OpenAI/Groq) to achieve high relevance and low latency (**\<2s** per query). The system uses **Supabase PostgreSQL** for session memory and is deployed on **GCP** with **Kubernetes** for orchestration, monitored via **Prometheus** and **Grafana**.

**Objectives:**

  * Deliver accurate, personalized recommendations using **hybrid retrieval**.
  * Ensure **scalability** for high query volumes.
  * Provide a modular, maintainable codebase for research and production.
  * Enable **observability** through comprehensive monitoring.

**Current Status (October 03, 2025):**

  * Developed on branch `feat/production-flask-api`.
  * Core functionality (ingestion, retrieval, API) complete.
  * Supabase connectivity stabilized with **IPv4/IPv6 fallback**.
  * Live deployment: [http://34.14.203.32:8080/](http://34.14.203.32:8080/).

-----

## Core Features

  * **Hybrid Retrieval**: Combines **BM25** (keyword-based) and **vector search** (semantic) with neural reranking (**BAAI/bge-reranker-base**).
  * **Session Memory**: Stores user interaction history in **Supabase PostgreSQL** for **multi-turn personalization**.
  * **Vector Storage**: **Astra DB** for efficient embedding storage and retrieval.
  * **Production API**: **Flask**-based endpoints for recommendations and health checks.
  * **Cloud-Native Deployment**: **Dockerized** application orchestrated with **Kubernetes** on **GCP**.
  * **Monitoring**: **Prometheus** for metrics collection, **Grafana** for visualization.
  * **Scalability**: **Horizontal Pod Autoscaler (HPA)** for dynamic scaling.
  * **Modularity**: Independent modules for ingestion, retrieval, and generation.

-----

## System Architecture

PerfectPick follows a layered, microservices-inspired architecture:

  * **Frontend Layer**: Minimal HTML/CSS interface for user queries and results.
  * **Backend Layer**: **Flask API** handling requests and orchestrating retrieval/generation.
  * **Data Layer**: Processes and validates Flipkart CSV data.
  * **Vector Store Layer**: **Astra DB** for storing product embeddings.
  * **Session Layer**: **Supabase PostgreSQL** for session persistence.
  * **Monitoring Layer**: **Prometheus** and **Grafana** for system observability.
  * **Deployment Layer**: **Docker** containers managed by **Kubernetes** on **GCP**.

-----

## Directory Structure

```
bhupencoD3-PerfectPick/
├── app.py
├── main.py
├── requirements.txt
├── Dockerfile
├── .env.example
├── data/
│   └── Flipkart_Mobiles_cleaned.csv
├── perfectpick/
│   ├── config.py
│   ├── data_converter.py
│   ├── data_ingestion.py
│   ├── retrieval.py
│   ├── generation.py
│   ├── recommender.py
│   ├── service.py
│   ├── session_memory.py
├── utils/
│   ├── logger.py
│   ├── custom_exception.py
│   ├── validators.py
├── templates/
│   └── index.html
├── static/
│   └── style.css
├── tests/
│   ├── test_config.py
│   ├── test_data_converter.py
│   ├── test_data_ingestion.py
│   ├── test_recommender.py
│   ├── test_service.py
├── notebooks/
│   └── exploration.ipynb
├── docs/
│   └── index.html
├── deployment/
│   ├── k8s/
│   │   ├── deployment.yml
│   │   ├── service.yml
│   │   ├── hpa.yml
│   ├── prometheus/
│   │   └── prometheus.yml
│   ├── grafana/
│   │   ├── dashboards/
│   │   ├── datasource.yml
│   ├── deploy-all.sh
│   ├── entrypoint.sh
│   └── SETUP.md
```

-----

## Component-Level Breakdown

### Core Application Modules

  * **`config.py`**

      * **Purpose**: Loads and validates environment variables from `.env`.
      * **Features**: Validates `DB_URL`, API keys; logs initialization status.
      * **Dependencies**: `os`, `python-dotenv`, `logging`.

  * **`data_converter.py`**

      * **Purpose**: Transforms raw product data into embedding-ready format.
      * **Features**: Normalizes prices, cleans text fields.
      * **Dependencies**: `pandas`.

  * **`data_ingestion.py`**

      * **Purpose**: Loads CSV data and indexes embeddings in **Astra DB**.
      * **Features**: Processes 3903 rows; uses **LangChain AstraDBVectorStore** (collection: `flipkart_recommendation`); Hugging Face embeddings (**BAAI/bge-base-en-v1.5**). Each product is vectorized based on a concatenated string of its key features (`model`, `RAM`, `storage`, `camera`, `processor`). The full product metadata is stored alongside the vector.
      * **Dependencies**: `pandas`, `langchain_astradb`, `sentence-transformers`.

  * **`retrieval.py`**

      * **Purpose**: Implements **hybrid retrieval** (**BM25 + vector search**).
      * **Features**: **BM25 index** (3917 documents) for keyword-based matches; vector search (top-20) for **semantic intent** (e.g., "good battery life"); neural reranking (**BAAI/bge-reranker-base**) to re-score the top-40 combined results, yielding a final **top-5** product set. Price filtering segments the data (Budget: ₹0-15K, Mid-range: ₹15-30K, Premium: ₹30-70K, Flagship: ₹70K+). The initial hybrid score is calculated as: $\text{Score}_{\text{hybrid}}(q, d) = \alpha \cdot \text{Score}_{\text{BM25}}(q, d) + (1-\alpha) \cdot \text{Score}_{\text{BGE}}(q, d)$, with $\alpha = 0.3$.
      * **Dependencies**: `rank_bm25`, `FlagEmbedding`, `langchain`.

  * **`generation.py`**

      * **Purpose**: Generates natural language responses using **LLMs**.
      * **Features**: Integrates **OpenAI/Groq** with Retrieval-Augmented Generation (**RAG**). The LLM uses a system prompt to enforce a persona ("Expert Mobile Consultant") and a strict output format. **Groq** is prioritized for low-latency inference, with **OpenAI** as a high-quality fallback.
      * **Dependencies**: `openai`, `groq`.

  * **`recommender.py`**

      * **Purpose**: Orchestrates retrieval and generation for recommendations.
      * **Features**: The module follows a clear RAG workflow: 1. **Retrieve Session** history. 2. **Hybrid Search** for candidate products. 3. **Rerank** candidates. 4. **Contextualize** by assembling the final RAG prompt, including the current query, session memory, and the top-5 documents. 5. **Generate** the response via LLM. 6. **Store Session** for personalization.
      * **Dependencies**: internal modules (`retrieval`, `generation`, `session_memory`).

  * **`service.py`**

      * **Purpose**: Defines **Flask API endpoints** (`/recommend`, `/health`, `/metrics`).
      * **Features**: Handles JSON requests, returns structured responses.
      * **Dependencies**: `flask`.

  * **`session_memory.py`**

      * **Purpose**: Manages user session history in **Supabase PostgreSQL**.
      * **Features**: Uses `psycopg2` client with IPv4/IPv6 fallback for reliability. The `session_memory` table stores the conversational context as a **JSONB** object, enabling **multi-turn recommendations** by providing the LLM with past queries and results for context-aware reasoning.
      * **Dependencies**: `psycopg2`, `collections.deque`, `socket`.

### Utility Layer

  * **`logger.py`**

      * **Purpose**: Provides **structured JSON logging**.
      * **Features**: Timestamps, log levels, JSON formatting.
      * **Dependencies**: `logging`, `json`.

  * **`custom_exception.py`**

      * **Purpose**: Defines custom exceptions for consistent error handling.
      * **Features**: Structured error messages for debugging.
      * **Dependencies**: None.

  * **`validators.py`**

      * **Purpose**: Validates input data and API requests.
      * **Features**: Checks query format, data integrity.
      * **Dependencies**: None.

### API and Frontend

  * **`app.py`**

      * **Purpose**: Main **Flask** application entrypoint.
      * **Features**: Initializes routes, logging, services; handles `/recommend`, `/health`, `/metrics`.
      * **Dependencies**: `flask`, internal modules.

  * **`templates/index.html`**

      * **Purpose**: Basic **HTML frontend** for user interaction.
      * **Features**: Query input, recommendation display.
      * **Dependencies**: None.

  * **`static/style.css`**

      * **Purpose**: Styles frontend interface.
      * **Features**: Minimal, responsive design.
      * **Dependencies**: None.

### Data and Experiments

  * **`data/Flipkart_Mobiles_cleaned.csv`**

      * **Purpose**: Source dataset with **3903 products** (8 columns: `model`, `price`, `RAM`, `storage`, `camera`, `battery`, `display`, `processor`).
      * **Features**: Cleaned, UTF-8 encoded.

  * **`notebooks/exploration.ipynb`**

      * **Purpose**: Exploratory data analysis and model prototyping.
      * **Features**: Visualizations, embedding experiments.
      * **Dependencies**: `jupyter`, `pandas`, `matplotlib`.

  * **`docs/index.html`**

      * **Purpose**: HTML documentation for contributors.
      * **Features**: API specs, module descriptions.

### Testing Suite

  * **Purpose**: Validates module functionality and integration.
  * **Files**:
      * `test_config.py`: Tests environment variable loading.
      * `test_data_converter.py`: Validates data preprocessing.
      * `test_data_ingestion.py`: Ensures CSV loading and indexing.
      * `test_recommender.py`: Tests recommendation pipeline.
      * `test_service.py`: Verifies API endpoints.
  * **Dependencies**: `pytest`.

### Deployment Layer

  * **Purpose**: Configures production deployment and monitoring.
  * **Files**:
      * `Dockerfile`: Builds application image (**Python 3.12**, non-root user).
      * `deploy-all.sh`: Automates **Kubernetes** deployment.
      * `entrypoint.sh`: Container startup script.
      * `k8s/deployment.yml`: Defines **3-replica deployment**.
      * `k8s/service.yml`: **LoadBalancer** for API access.
      * `k8s/hpa.yml`: Autoscales pods (**3-10 replicas**, **50% CPU**).
      * `prometheus/prometheus.yml`: Configures metrics scraping.
      * `grafana/datasource.yml`: Links Grafana to Prometheus.
      * `grafana/dashboards/`: Prebuilt dashboards for CPU, memory, latency.
      * `SETUP.md`: Deployment instructions.

-----

## Technology Stack

| Category | Technology | Purpose |
| :--- | :--- | :--- |
| **API** | **Flask** | RESTful endpoints |
| **Data Storage** | **Supabase PostgreSQL** | Session memory |
| **Vector Storage** | **Astra DB** | Product embeddings |
| **Retrieval Models** | **BM25**, **BGE Embeddings** (`BAAI/bge-base-en-v1.5`), **BGE Reranker** (`BAAI/bge-reranker-base`) | Hybrid retrieval and reranking |
| **Generation LLMs** | **OpenAI**, **Groq** | LLM integration for response generation |
| **Containerization** | **Docker** (BuildKit-enabled) | Application packaging |
| **Orchestration** | **Kubernetes** (**GKE**) | Container management |
| **Cloud** | **Google Cloud Platform** (GCE, GKE, Secret Manager) | Hosting and infrastructure |
| **Monitoring** | **Prometheus**, **Grafana** | Metrics collection and visualization |
| **Testing** | **Pytest** | Unit and integration tests |
| **Utilities** | `python-dotenv`, `pandas`, `sentence-transformers` | Configuration, data manipulation, embedding utilities |

-----

## Setup and Local Development

**Prerequisites:**

  * **OS**: Arch Linux (or Ubuntu/Debian)
  * **Python**: **3.12+** (use `pyenv`)
  * **Docker**: Install via `sudo pacman -S docker`
  * **Git**: For repository cloning

**Setup Steps:**

1.  **Clone**: `git clone <repo-url>; cd bhupencoD3-PerfectPick`
2.  **Virtual Env**: `python -m venv venv; source venv/bin/activate; pip install -r requirements.txt`
3.  **Configure `.env`**:
    ```dotenv
    DB_URL=postgresql://postgres.lxbououtadfxarleksun:<password>@aws-1-ap-southeast-1.pooler.supabase.com:6543/postgres?sslmode=require
    OPENAI_API_KEY=sk-proj-...
    GROQ_API_KEY=gsk_...
    ASTRA_DB_API_ENDPOINT=https://ad6829b0-39f3-43aa-9a13-43036ed6bed2-us-east-2.apps.astra.datastax.com
    ASTRA_DB_APPLICATION_TOKEN=AstraCS:...
    ASTRA_DB_KEYSPACE=default_keyspace
    HF_TOKEN=hf_qtSBHF...
    DATA_FILE_PATH=data/Flipkart_Mobiles_cleaned.csv
    FLASK_ENV=development
    FLASK_PORT=8000
    ```
4.  **Run**: `python app.py`
5.  **Test**: `curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d '{"query": "best budget phone"}'`

**Docker Setup:**

1.  **Build**: `export DOCKER_BUILDKIT=1; docker build -t perfectpick:latest .`
2.  **Run**: `docker run --env-file .env -p 8000:8000 --name perfectpick perfectpick:latest`
3.  **Compose**: `docker-compose up --build` (using `docker-compose.yml`)

-----

## Deployment Workflow

**Overview:**
PerfectPick is deployed on **GCP** using **Google Kubernetes Engine (GKE)** for orchestration, **Artifact Registry** for images, **Secret Manager** for credentials, and **Cloud Build** for CI/CD. The live instance is accessible at **[http://34.14.203.32:8080/](http://34.14.203.32:8080/)**.

**Steps:**

1.  **Enable GCP APIs**: GKE, Cloud Build, Artifact Registry, Secret Manager.
2.  **Create GKE Cluster**:
    ```bash
    gcloud container clusters create perfectpick-cluster --zone us-central1-a --machine-type e2-medium --num-nodes 3 --enable-ip-alias --enable-autoscaling --min-nodes 1 --max-nodes 5
    ```
3.  **Store Secrets**:
    ```bash
    gcloud secrets create db-url --data-file=<file>
    gcloud secrets create openai-key --data-file=<file>
    ```
4.  **Build and Push Image**:
    ```bash
    gcloud auth configure-docker
    docker tag perfectpick:latest gcr.io/<project-id>/perfectpick:latest
    docker push gcr.io/<project-id>/perfectpick:latest
    ```
5.  **Apply Kubernetes Manifests**:
    ```bash
    kubectl apply -f deployment/k8s/
    ```
6.  **Verify**:
    ```bash
    kubectl get pods
    kubectl port-forward service/perfectpick-service 8080:80
    ```
7.  **Access**: [http://34.14.203.32:8080/](http://34.14.203.32:8080/)
8.  **Scale**:
    ```bash
    kubectl scale deployment perfectpick --replicas=3
    kubectl apply -f deployment/k8s/hpa.yml
    ```

**Cloud Build CI/CD (`cloudbuild.yaml`):**

```yaml
steps:
- name: 'gcr.io/cloud-builders/docker'
  args: ['build', '-t', 'gcr.io/$PROJECT_ID/perfectpick:$COMMIT_SHA', '.']
- name: 'gcr.io/cloud-builders/docker'
  args: ['push', 'gcr.io/$PROJECT_ID/perfectpick:$COMMIT_SHA']
- name: 'gcr.io/cloud-builders/gke-deploy'
  args:
  - run
  - --filename=deployment/k8s/
  - --image=gcr.io/$PROJECT_ID/perfectpick:$COMMIT_SHA
  - --location=us-central1-a
  - --cluster=perfectpick-cluster
```

**Optional Cloud SQL Migration:**

1.  **Create**: `gcloud sql instances create perfectpick-db --database-version=POSTGRES_15 --tier=db-f1-micro`
2.  **Update `DB_URL`**: `postgresql://postgres:<pass>@<cloud-sql-ip>:5432/postgres`
3.  **Migrate**: `pg_dump <supabase-url> | gcloud sql import sql perfectpick-db -`

-----

## Monitoring and Observability

### Prometheus

  * **Enabled**: `gcloud container clusters update perfectpick-cluster --monitoring=SYSTEM,WORKLOAD`
  * **Config (`deployment/prometheus/prometheus.yml`)**:
    ```yaml
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'perfectpick'
      static_configs:
      - targets: ['perfectpick-service:8000']
      metrics_path: /metrics
    ```

### Grafana

  * **Deployment (`deployment/grafana/deployment.yml`)**:
    ```yaml
    apiVersion: apps/v1
    kind: Deployment
    metadata:
      name: grafana
    spec:
      replicas: 1
      selector:
        matchLabels:
          app: grafana
      template:
        metadata:
          labels:
            app: grafana
        spec:
          containers:
          - name: grafana
            image: grafana/grafana:latest
            ports:
            - containerPort: 3000
            env:
            - name: GF_SECURITY_ADMIN_PASSWORD
              value: "admin"
    ---
    apiVersion: v1
    kind: Service
    metadata:
      name: grafana-service
    spec:
      type: LoadBalancer
      ports:
      - port: 3000
      selector:
        app: grafana
    ```
  * **Access**: `<load-balancer-ip>:3000`
  * **Data Source**: Prometheus (`http://prometheus-operated:9090`)
  * **Dashboards**: CPU usage, memory, query latency, DB connection errors, **RAG performance breakdown**.

### Custom Metrics (`app.py`)

PerfectPick tracks key metrics for the RAG pipeline to ensure recommendation quality and efficiency:

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from flask import Response

# General Metrics
QUERY_COUNTER = Counter('perfectpick_queries_total', 'Total queries')
LATENCY_HISTOGRAM = Histogram('perfectpick_query_duration_seconds', 'Query latency')
SESSION_HITS = Counter('perfectpick_session_memory_retrievals', 'Number of times session memory was retrieved')

# RAG & LLM Specific Metrics
RETRIEVAL_LATENCY = Histogram('perfectpick_retrieval_duration_seconds', 'Duration of hybrid retrieval stage')
RERANKING_TIME = Histogram('perfectpick_reranking_duration_seconds', 'Duration of neural reranking stage')
LLM_PROVIDER_COUNTER = Counter('perfectpick_llm_provider_used', 'Count of responses by LLM provider', ['provider'])
LLM_TOKEN_USAGE = Gauge('perfectpick_llm_tokens_used', 'Tokens used per query (Input/Output)', ['type', 'provider'])

@app.route('/metrics')
def metrics():
    return Response(generate_latest(), mimetype='text/plain')

@app.route('/recommend', methods=['POST'])
def recommend():
    # ... logic for retrieval, reranking, and generation ...
    with LATENCY_HISTOGRAM.time():
        QUERY_COUNTER.inc()
        # RETRIEVAL_LATENCY and RERANKING_TIME are observed internally
        # LLM_PROVIDER_COUNTER and LLM_TOKEN_USAGE are set after generation
        return jsonify(results)
```

### Alerts

Critical alerts are configured to monitor both system stability and the quality of the AI service:

  * **System Health**: CPU $>$80%, Latency $>$2s (P95), Pod Restarts $>$3/hr.
  * **Quality Degradation**: DB errors $>$5/min (loss of personalization), **Reranker Latency $>$100ms** (RAG bottleneck), **LLM API Failures $>$2%** (critical service interruption).
  * Configured via **Grafana notifications** (Slack/Email).

-----

## Scalability and Performance

  * **Horizontal Scaling**: **HPA** scales pods (**3-10**) based on **50% CPU** utilization.
  * **Performance**: Retrieval **\<1s**, total query **\<2s**; model initialization **\~3-5min** (CPU).
  * **Optimization**: Pre-downloaded BGE models; cached embeddings in **Astra DB**.
  * **Load Handling**: Tested for **100 concurrent users** with **\<5% latency increase**.

-----

## Future Improvements

  * **CI/CD**: Integrate **GitHub Actions** with **Cloud Build** for automated testing, image building, and **Canary Deployments** for new LLM/retrieval models.
  * **Caching**: Implement **Redis** for both **exact query caching** (for popular queries) and **session caching** (to reduce load on Supabase).
  * **Multi-Modal Search**: Incorporate image-based search with **CLIP** embeddings to allow for visual queries (e.g., "find a phone that looks like this").
  * **A/B Testing**: Implement a solution using Kubernetes **Ingress/Service Mesh** to A/B test different retrieval strategies (e.g., varying the $\alpha$ value in hybrid scoring, or testing new rerankers).
  * **Authentication**: Add **OAuth2** for secure API access.
  * **Multi-Product Support**: Extend to other Flipkart categories (e.g., laptops).
  * **Personalization**: Explore **federated learning** for user-specific models.

-----

## License

This project is licensed under the **MIT License**.