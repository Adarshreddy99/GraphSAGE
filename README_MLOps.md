# 🚀 MLOps Complete Reference Guide
### Git · DVC · MLflow · Docker — Everything You Need, In Order

---

## Table of Contents

1. [The Big Picture — How Everything Connects](#-the-big-picture--how-everything-connects)
2. [Part 1 — Git: Version Control](#-part-1--git-version-control)
3. [Part 2 — DVC: Data Version Control](#-part-2--dvc-data-version-control)
4. [Part 3 — MLflow: Experiment Tracking](#-part-3--mlflow-experiment-tracking)
5. [Part 4 — Docker: Containerization](#-part-4--docker-containerization)
6. [The Full Workflow — End to End](#-the-full-workflow--end-to-end)
7. [Docker Common Issues & Solutions](#-docker-common-issues--solutions)

---

## 🗺 The Big Picture — How Everything Connects

Before diving into each tool, here's how they work together:

```
Your Codebase
     │
     ▼
  [ Git ]  ──── tracks code changes, commits, branches
     │
     ├── [ DVC ]  ──── tracks large data files and pipeline stages
     │                 (Git tracks the DVC pointer files)
     │
     ├── [ MLflow ] ── tracks experiment runs, metrics, and model artifacts
     │                 (Git tracks your training scripts)
     │
     └── [ Docker ] ── packages everything (code + env + dependencies)
                       into a portable, reproducible container
```

Think of it this way:
- **Git** remembers every version of your code.
- **DVC** remembers every version of your data and how you processed it.
- **MLflow** remembers every training experiment you ran and what results you got.
- **Docker** ensures that the code + data + environment run the same way everywhere.

---

## 📁 Part 1 — Git: Version Control

### What Git Does

Git tracks changes in text-based files (`.py`, `.yaml`, `.txt`, `.json`, etc.). It creates a complete history of your project so you can go back to any point in time.

### Core Concepts

| Concept | What It Is |
|---|---|
| **Repository (repo)** | The folder Git is tracking |
| **Working Directory** | Your actual files on disk |
| **Staging Area (Index)** | A "prep zone" where you decide what goes into the next snapshot |
| **Commit** | A permanent snapshot of the staging area |
| **Branch** | A parallel version of the repo |
| **Remote** | A copy of the repo hosted online (GitHub, GitLab) |

### Setting Up Git

```bash
# Install Git (Ubuntu/Debian)
sudo apt install git

# Configure your identity (do this once globally)
git config --global user.name "Your Name"
git config --global user.email "you@example.com"

# Initialise a new repository in the current folder
git init

# Or clone an existing one from remote
git clone https://github.com/your-username/your-repo.git
```

### The Core Daily Workflow

```bash
# 1. Check the current state of your working directory
git status

# 2. See what changed line by line
git diff

# 3. Stage specific files for the next commit
git add src/train.py
git add requirements.txt

# Stage ALL changed files at once (use carefully)
git add .

# 4. Commit the staged snapshot with a message
git commit -m "feat: add FAISS retrieval to recommender pipeline"

# 5. Push commits to the remote branch
git push origin main

# 6. Pull the latest changes from remote (always do this before you start working)
git pull origin main
```

### Branching — Working on Features Without Breaking Main

```bash
# Create a new branch and switch to it
git checkout -b feature/add-mlflow-tracking

# List all branches (* marks your current one)
git branch

# Switch to an existing branch
git checkout main

# Merge a feature branch into main
git checkout main
git merge feature/add-mlflow-tracking

# Delete the branch after merging
git branch -d feature/add-mlflow-tracking

# Push a new branch to remote for the first time
git push -u origin feature/add-mlflow-tracking
```

### Undoing Mistakes

```bash
# Undo staging a file (unstage it, keep the changes on disk)
git restore --staged src/train.py

# Discard changes in a file entirely (go back to last commit)
git restore src/train.py

# Undo the last commit but KEEP changes staged
git reset --soft HEAD~1

# Undo the last commit and UNSTAGE changes (changes still on disk)
git reset --mixed HEAD~1

# Undo the last commit and DELETE all changes (dangerous — irreversible)
git reset --hard HEAD~1

# Revert a specific commit by creating an "anti-commit"
# (safer for shared branches — doesn't rewrite history)
git revert <commit-hash>
```

### The `.gitignore` File

This file tells Git which files to completely ignore. Always create this before your first commit.

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/
*.egg-info/

# Secrets — NEVER commit these
.env
credentials.json
service_account.json
*.pem
*.key

# DVC and data (DVC handles these separately)
data/raw/
data/processed/
models/
*.dvc/cache

# MLflow
mlruns/

# Docker
.dockerignore

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
```

### Handling Merge Conflicts

A merge conflict happens when you and a teammate edit the same line of the same file and Git doesn't know which version to keep.

```bash
# Pull triggers the conflict
git pull origin main

# Git marks the conflicting file like this:
# <<<<<<< HEAD
# your version of the code
# =======
# their version of the code
# >>>>>>> origin/main

# Open the file, delete the conflict markers, keep the correct code.
# Then:
git add src/model.py
git commit -m "fix: resolve merge conflict in model.py"
git push origin main
```

### Viewing History

```bash
# See full commit log
git log

# Compact one-line view
git log --oneline

# Graphical branch view in terminal
git log --oneline --graph --all

# See what changed in a specific commit
git show <commit-hash>

# Compare two branches
git diff main feature/add-mlflow-tracking
```

### Secret Scanning Warning

> ⚠️ **GitHub blocks pushes that contain API keys or OAuth secrets.**
> If you accidentally commit a secret:
> ```bash
> # Step 1: Undo the commit (keep changes staged)
> git reset --soft HEAD~1
> # Step 2: Remove the secret from the file
> # Step 3: Re-commit without the secret
> git commit -m "fix: remove accidentally committed credentials"
> # Step 4: Rotate/invalidate the leaked secret immediately
> ```
> Always use `.env` files for secrets and ensure `.env` is in `.gitignore`.

---

## 📦 Part 2 — DVC: Data Version Control

### What DVC Does

Git cannot efficiently store large binary files (datasets, model weights, embeddings). DVC solves this by:
1. Storing a tiny **pointer file** (`.dvc`) in Git.
2. Storing the **actual large file** in a remote storage (Google Drive, S3, GCS, Azure).

This way, Git stays fast and light, but you still get full data versioning.

### Installing DVC

```bash
pip install dvc

# Install with specific remote support
pip install dvc[gdrive]   # Google Drive
pip install dvc[s3]       # Amazon S3
pip install dvc[gs]       # Google Cloud Storage
pip install dvc[azure]    # Azure Blob Storage
```

### Initialising DVC in Your Project

```bash
# Must be done inside a Git repo
git init
dvc init

# DVC creates these files — commit them to Git
git add .dvc/config .dvc/.gitignore .dvcignore
git commit -m "chore: initialise DVC"
```

### Adding Data Files to DVC Tracking

```bash
# Track a file with DVC
dvc add data/raw/papers.csv

# What this does internally:
# 1. Moves papers.csv to .dvc/cache (content-addressed storage)
# 2. Creates a pointer file: data/raw/papers.csv.dvc
# 3. Adds papers.csv to .gitignore so Git ignores the real file

# Add the pointer file to Git (NOT the data file)
git add data/raw/papers.csv.dvc data/raw/.gitignore
git commit -m "data: track raw papers dataset with DVC"
```

### Setting Up a Remote Storage

```bash
# Add a Google Drive folder as remote
dvc remote add -d myremote gdrive://<folder-id>

# Add an S3 bucket as remote
dvc remote add -d myremote s3://my-bucket/dvc-store

# Save the remote config to Git
git add .dvc/config
git commit -m "chore: configure DVC remote storage"
```

### Pushing and Pulling Data

```bash
# Push all tracked data to the remote storage
dvc push

# Pull data from remote (after cloning the repo on a new machine)
dvc pull

# Pull only a specific file
dvc pull data/raw/papers.csv.dvc
```

### DVC Pipelines — The Real Power

DVC pipelines let you define your entire ML workflow as stages. DVC then intelligently skips stages whose inputs haven't changed.

**`dvc.yaml` — the pipeline definition file:**

```yaml
stages:
  data_prep:
    cmd: python src/data/preprocess.py
    deps:
      - src/data/preprocess.py
      - data/raw/papers.csv
    outs:
      - data/processed/papers_clean.csv

  featurise:
    cmd: python src/features/build_features.py
    deps:
      - src/features/build_features.py
      - data/processed/papers_clean.csv
    outs:
      - data/features/embeddings.npy

  train:
    cmd: python src/models/train.py
    deps:
      - src/models/train.py
      - data/features/embeddings.npy
    params:
      - params.yaml:
        - model.learning_rate
        - model.epochs
    outs:
      - models/recommender.pkl
    metrics:
      - reports/metrics.json:
          cache: false

  evaluate:
    cmd: python src/models/evaluate.py
    deps:
      - src/models/evaluate.py
      - models/recommender.pkl
    metrics:
      - reports/evaluation.json:
          cache: false
```

**`params.yaml` — your hyperparameters:**

```yaml
model:
  learning_rate: 0.001
  epochs: 50
  batch_size: 32
  embedding_dim: 128
```

### Running the Pipeline

```bash
# Run ALL stages (only re-runs what changed)
dvc repro

# Force re-run everything, ignoring cache
dvc repro --force

# Run only a specific stage
dvc repro train

# See what stages WOULD run without actually running them (dry run)
dvc repro --dry

# Visualise the pipeline DAG
dvc dag
```

### Comparing Experiments with DVC

```bash
# Show current metrics
dvc metrics show

# Compare metrics between Git commits or branches
dvc metrics diff main feature/new-model

# Compare parameters
dvc params diff main feature/new-model
```

### The Full DVC Commit Cycle

```bash
# After running dvc repro:
# 1. Push the new data outputs to remote storage
dvc push

# 2. Commit the updated lock file and pointer files to Git
git add dvc.lock data/processed/.gitignore models/.gitignore
git commit -m "experiment: retrain with lr=0.001, epochs=50"
git push origin main
```

### Reproducing on a New Machine

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install -r requirements.txt

# Pull all data from DVC remote
dvc pull

# Re-run the pipeline (DVC uses cache, so only changed stages run)
dvc repro
```

---

## 📊 Part 3 — MLflow: Experiment Tracking

### What MLflow Does

MLflow tracks your ML experiments. Every time you train a model, MLflow records:
- **Parameters**: hyperparameters you used.
- **Metrics**: accuracy, loss, NDCG, etc.
- **Artifacts**: the saved model file, plots, confusion matrices.
- **Tags**: metadata like dataset version, Git commit hash.

This lets you compare dozens of runs and find the best model.

### Installing MLflow

```bash
pip install mlflow
```

### Starting the MLflow UI

```bash
# Start the tracking server (runs at http://localhost:5000)
mlflow ui

# Or specify a custom port
mlflow ui --port 5001

# Point to a specific tracking folder
mlflow ui --backend-store-uri ./mlruns
```

### Logging an Experiment in Python

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

# Tell MLflow where to log (default is ./mlruns)
mlflow.set_tracking_uri("http://localhost:5000")

# Set the experiment name (creates it if it doesn't exist)
mlflow.set_experiment("paper-recommender-v1")

with mlflow.start_run(run_name="random-forest-baseline"):

    # --- Log parameters ---
    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.01,
        "embedding_dim": 128
    }
    mlflow.log_params(params)

    # --- Train your model ---
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # --- Log metrics ---
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("ndcg_at_10", ndcg)

    # Log metrics across training steps
    for epoch, loss in enumerate(training_losses):
        mlflow.log_metric("train_loss", loss, step=epoch)

    # --- Log the model ---
    mlflow.sklearn.log_model(model, artifact_path="model")

    # --- Log arbitrary files ---
    mlflow.log_artifact("reports/evaluation_plot.png")
    mlflow.log_artifact("data/processed/sample.csv", artifact_path="data-sample")

    # --- Log tags ---
    mlflow.set_tag("git_commit", "a3f9c12")
    mlflow.set_tag("dataset_version", "v2.1")
    mlflow.set_tag("developer", "your-name")
```

### Autologging — Zero-Code Tracking

MLflow can automatically log parameters and metrics for popular frameworks:

```python
# For scikit-learn
mlflow.sklearn.autolog()

# For PyTorch Lightning
mlflow.pytorch.autolog()

# For TensorFlow/Keras
mlflow.tensorflow.autolog()

# For XGBoost
mlflow.xgboost.autolog()

# For LightGBM
mlflow.lightgbm.autolog()
```

### Querying Runs Programmatically

```python
import mlflow

client = mlflow.MlflowClient()

# Get all experiments
experiments = client.search_experiments()

# Search runs with filters
runs = mlflow.search_runs(
    experiment_names=["paper-recommender-v1"],
    filter_string="metrics.accuracy > 0.85",
    order_by=["metrics.accuracy DESC"]
)

# Get the best run
best_run = runs.iloc[0]
print(f"Best run ID: {best_run['run_id']}")
print(f"Best accuracy: {best_run['metrics.accuracy']}")
```

### Loading a Logged Model

```python
import mlflow.sklearn

# Load a specific run's model
model = mlflow.sklearn.load_model(f"runs:/<run_id>/model")

# Or load from the MLflow Model Registry
model = mlflow.sklearn.load_model("models:/paper-recommender/Production")
```

### MLflow Model Registry (Promoting Models)

```python
# Register a model from a run
result = mlflow.register_model(
    model_uri=f"runs:/<run_id>/model",
    name="paper-recommender"
)

# Transition model stages via client
client = mlflow.MlflowClient()

client.transition_model_version_stage(
    name="paper-recommender",
    version=result.version,
    stage="Staging"  # or "Production" or "Archived"
)
```

### Connecting MLflow with DVC

A good pattern: use DVC to version data and pipeline stages, and MLflow to log the results of each `dvc repro` run.

```python
# Inside your train.py stage script
import mlflow
import dvc.api

# Read params from DVC params.yaml
params = dvc.api.params_show()

with mlflow.start_run():
    mlflow.log_params(params["model"])
    # ... train, evaluate, log metrics ...
```

### Storing MLflow Remotely

By default, MLflow stores everything in a local `./mlruns` folder. For team use:

```bash
# Use an SQLite database as backend
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts \
    --host 0.0.0.0 \
    --port 5000

# Use PostgreSQL + S3 for production
mlflow server \
    --backend-store-uri postgresql://user:pass@host/mlflow_db \
    --default-artifact-root s3://your-bucket/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

---

## 🐳 Part 4 — Docker: Containerization

### What Docker Does

Docker packages your application — code, Python version, all libraries, OS tools — into a single self-contained "container." If it runs on your machine, it runs on any server, any cloud, any teammate's laptop.

### Core Concepts

| Concept | Analogy | Description |
|---|---|---|
| **Image** | A blueprint / DVD | Static, read-only, built from a Dockerfile |
| **Container** | A running instance | A live process launched from an image |
| **Dockerfile** | A recipe | Step-by-step instructions to build an image |
| **docker-compose.yml** | An orchestrator | Defines and starts multiple containers together |
| **Registry** | A library | Stores and shares images (Docker Hub, ECR, GCR) |
| **Volume** | An external hard drive | Persistent storage attached to a container |
| **Network** | A private LAN | How containers talk to each other |

### The Dockerfile — Building an Image

```dockerfile
# ─────────────────────────────────────────────
# STAGE 1: Builder
# Installs heavy build dependencies (compilers, etc.)
# ─────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system-level build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ─────────────────────────────────────────────
# STAGE 2: Runtime
# Copies only the compiled result — much smaller image
# ─────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY params.yaml .
COPY models/ ./models/

# Fix Windows line endings in shell scripts
RUN find ./scripts -name "*.sh" -exec dos2unix {} \; \
    && chmod +x ./scripts/*.sh

# Set Python path so `src` is importable as a package
ENV PYTHONPATH=/app/src:/app

# CRITICAL: Use 0.0.0.0, not 127.0.0.1
# 127.0.0.1 = "talk only to yourself inside the container"
# 0.0.0.0   = "listen to the outside world"
ENV HOST=0.0.0.0
ENV PORT=8000

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Building and Running Images

```bash
# Build an image and tag it
docker build -t my-recommender:latest .

# Build without using any cache (clean build)
docker build --no-cache -t my-recommender:latest .

# Build targeting a specific stage
docker build --target builder -t my-recommender:build-stage .

# Run a container from the image
docker run -p 8000:8000 my-recommender:latest

# Run in detached mode (background)
docker run -d -p 8000:8000 --name recommender-api my-recommender:latest

# Run with environment variables
docker run -d -p 8000:8000 \
    -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
    -e MODEL_VERSION=v2 \
    my-recommender:latest

# Run with a volume (mount local folder into container)
docker run -d -p 8000:8000 \
    -v $(pwd)/models:/app/models \
    my-recommender:latest

# Run interactively (drop into a shell inside the container)
docker run -it --entrypoint /bin/bash my-recommender:latest
```

### Managing Containers

```bash
# List running containers
docker ps

# List ALL containers (including stopped ones)
docker ps -a

# View logs of a running container
docker logs recommender-api

# Follow logs in real time
docker logs -f recommender-api

# Stop a container gracefully
docker stop recommender-api

# Kill immediately
docker kill recommender-api

# Remove a stopped container
docker rm recommender-api

# Stop and remove in one command
docker rm -f recommender-api

# Execute a command inside a running container (for debugging)
docker exec -it recommender-api /bin/bash
docker exec -it recommender-api python -c "import faiss; print(faiss.__version__)"
```

### Managing Images

```bash
# List all local images
docker images

# Remove an image
docker rmi my-recommender:latest

# Remove ALL unused images (reclaim disk space)
docker image prune -a

# Remove ALL unused containers, images, networks, and volumes
docker system prune -a

# Push an image to Docker Hub
docker tag my-recommender:latest yourusername/my-recommender:latest
docker push yourusername/my-recommender:latest

# Pull an image from Docker Hub
docker pull yourusername/my-recommender:latest
```

### Docker Compose — Running Multiple Services

`docker-compose.yml` lets you define and start your entire system (API + MLflow + database) with a single command.

```yaml
version: "3.9"

services:

  # ─── FastAPI Application ───
  api:
    build:
      context: .
      dockerfile: Dockerfile
      target: runtime
    container_name: recommender-api
    ports:
      - "8000:8000"           # host:container
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
      - PYTHONPATH=/app/src:/app
      - MODEL_PATH=/app/models/recommender.pkl
    volumes:
      - ./models:/app/models   # live-mount models folder
      - ./logs:/app/logs
    depends_on:
      - mlflow
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ─── MLflow Tracking Server ───
  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    container_name: mlflow-server
    ports:
      - "5000:5000"
    environment:
      - MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow/mlflow.db
    volumes:
      - mlflow-data:/mlflow
    command: >
      mlflow server
      --backend-store-uri sqlite:///mlflow/mlflow.db
      --default-artifact-root /mlflow/artifacts
      --host 0.0.0.0
      --port 5000
    restart: unless-stopped

  # ─── Gradio / Frontend UI ───
  ui:
    build:
      context: .
      dockerfile: Dockerfile.ui
    container_name: recommender-ui
    ports:
      - "7860:7860"
    environment:
      - API_URL=http://api:8000
    depends_on:
      - api
    restart: unless-stopped

volumes:
  mlflow-data:
```

### Docker Compose Commands

```bash
# Start all services (builds if needed)
docker-compose up

# Start in detached/background mode
docker-compose up -d

# Force rebuild images before starting
docker-compose up --build

# Full clean rebuild (no cache)
docker-compose build --no-cache
docker-compose up -d

# Stop all services (keeps containers)
docker-compose stop

# Stop and REMOVE containers (keeps volumes and images)
docker-compose down

# Stop and remove containers AND volumes (data wipe)
docker-compose down -v

# View logs of a specific service
docker-compose logs api
docker-compose logs -f mlflow    # follow

# Restart a specific service
docker-compose restart api

# Run a one-off command in a service
docker-compose run --rm api python src/models/train.py

# Scale a service to multiple instances
docker-compose up --scale api=3
```

### Networking in Docker Compose

When services are in the same `docker-compose.yml`, they can reach each other by **service name** as the hostname:

```python
# Inside the 'api' container, reach mlflow like this:
MLFLOW_TRACKING_URI = "http://mlflow:5000"   # 'mlflow' = service name
# NOT http://localhost:5000 (that's inside the mlflow container itself)
```

```
  Your Browser   →  localhost:8000  →  [api container]
                                            │
                                            │  http://mlflow:5000
                                            ▼
                                      [mlflow container]
```

### Port Mapping Rules

```
"8080:8000"
  │      └── Port INSIDE the container
  └── Port on YOUR machine (host)

Visit localhost:8080 in your browser → Docker routes it to port 8000 inside the container.

If "Port already in use" error:
Change the LEFT port:  "9000:8000"  → visit localhost:9000
```

---

## 🔄 The Full Workflow — End to End

Here is the exact order you should initialise and run everything in a brand new project:

### Phase 1 — Project Setup

```bash
# 1. Create project folder
mkdir my-ml-project && cd my-ml-project

# 2. Initialise Git
git init
git branch -M main

# 3. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate         # Linux/Mac
# .venv\Scripts\activate          # Windows

# 4. Install all tools
pip install dvc[gdrive] mlflow fastapi uvicorn

# 5. Initialise DVC (must be inside a Git repo)
dvc init

# 6. Create folder structure
mkdir -p src/{data,features,models,api} data/{raw,processed,features} models reports scripts

# 7. Add __init__.py to every src folder (critical for Python imports)
touch src/__init__.py src/data/__init__.py src/features/__init__.py \
      src/models/__init__.py src/api/__init__.py

# 8. Create .gitignore (add all exclusions shown in Part 1)
# 9. Create requirements.txt with pinned versions
pip freeze > requirements.txt

# 10. First Git commit
git add .
git commit -m "chore: initial project structure"

# 11. Connect to GitHub
git remote add origin https://github.com/your-username/your-repo.git
git push -u origin main

# 12. Add DVC remote
dvc remote add -d myremote gdrive://<folder-id>
git add .dvc/config
git commit -m "chore: configure DVC remote"
git push
```

### Phase 2 — Adding Data

```bash
# Track your dataset with DVC
dvc add data/raw/papers.csv

# Commit the pointer file
git add data/raw/papers.csv.dvc data/raw/.gitignore
git commit -m "data: add raw papers dataset v1"

# Push data to remote storage
dvc push

# Push commit to GitHub
git push origin main
```

### Phase 3 — Build the Pipeline

```bash
# Create dvc.yaml (define your stages as shown in Part 2)
# Create params.yaml (define your hyperparameters)

# Run the pipeline
dvc repro

# After successful run, push data outputs
dvc push

# Commit pipeline definition and lock file
git add dvc.yaml dvc.lock params.yaml
git commit -m "pipeline: define data prep, train, evaluate stages"
git push origin main
```

### Phase 4 — Experiment with MLflow

```bash
# Start MLflow UI (in a separate terminal)
mlflow ui --port 5000

# Edit params.yaml to try new hyperparameters
# Run the pipeline (only changed stages re-execute)
dvc repro

# MLflow logs each run automatically (configure in train.py)

# After a good experiment:
dvc push
git add dvc.lock reports/
git commit -m "experiment: try lr=0.0001, accuracy improved to 0.89"
git push origin main
```

### Phase 5 — Package with Docker

```bash
# Create Dockerfile (as shown in Part 4)
# Create docker-compose.yml

# Build and test locally
docker-compose build
docker-compose up -d

# Test the API
curl http://localhost:8000/health
curl http://localhost:8000/recommend?query="graph neural networks"

# View logs
docker-compose logs -f api

# If everything works, tag and push the image
docker tag my-recommender:latest yourusername/my-recommender:v1.0
docker push yourusername/my-recommender:v1.0

# Commit Docker files
git add Dockerfile docker-compose.yml .dockerignore
git commit -m "docker: add production Dockerfile and compose config"
git push origin main
```

---

## 🚨 Docker Common Issues & Solutions

This section covers the most common errors you'll hit while building images or running containers, especially for FastAPI/ML projects.

---

### Issue 1 — `exec ./scripts/start.sh: no such file or directory`

**Cause:** Your `.sh` script was created on Windows with CRLF (`\r\n`) line endings. Linux sees the `\r` as part of the filename.

**Fix:**
```dockerfile
# In Dockerfile, add dos2unix to clean all scripts
RUN apt-get update && apt-get install -y dos2unix \
    && find ./scripts -name "*.sh" -exec dos2unix {} \; \
    && chmod +x ./scripts/*.sh
```

```bash
# Or fix locally before building
sed -i 's/\r//' scripts/start.sh
# Or use dos2unix directly
dos2unix scripts/start.sh
```

---

### Issue 2 — `ModuleNotFoundError: No module named 'src'`

**Cause:** Python can't find your `src` package. This happens when your `src/` folder is missing `__init__.py` or `PYTHONPATH` isn't set.

**Fix:**
```bash
# Ensure every subfolder has __init__.py
touch src/__init__.py src/data/__init__.py src/models/__init__.py
```

```dockerfile
# In Dockerfile, set PYTHONPATH explicitly
ENV PYTHONPATH=/app/src:/app
```

---

### Issue 3 — `EMPTY_RESPONSE` or browser can't reach the app

**Cause:** The app inside the container is bound to `127.0.0.1` (loopback), which means it only accepts connections from inside the container itself, not from your browser.

**Fix:** Always bind to `0.0.0.0` when running inside Docker.

```python
# FastAPI with Uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000)

# Gradio
demo.launch(server_name="0.0.0.0", server_port=7860)

# Flask
app.run(host="0.0.0.0", port=5000)
```

---

### Issue 4 — Docker uses old cached layer after updating `requirements.txt`

**Cause:** Docker caches every layer. If `requirements.txt` didn't change in the filesystem hash, Docker reuses the cached `pip install` layer.

**Fix:**
```bash
# Force a full rebuild ignoring all cache
docker-compose build --no-cache

# Or rebuild only a specific service
docker-compose build --no-cache api
```

**Best practice to avoid this in the Dockerfile:**
```dockerfile
# Copy requirements BEFORE the rest of the code
# This way, pip install only re-runs when requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# THEN copy your code (code changes don't trigger pip reinstall)
COPY src/ ./src/
```

---

### Issue 5 — Container exits with code 137 (OOM Kill)

**Cause:** The container ran out of memory. Code 137 = killed by the OS (SIGKILL). Common with PyTorch, FAISS, or large Transformers.

**Fix:**
- Open **Docker Desktop → Settings → Resources → Memory** → increase to 4GB+.
- Or set memory limits in `docker-compose.yml`:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4g
        reservations:
          memory: 2g
```

---

### Issue 6 — `Port is already allocated` or `address already in use`

**Cause:** Another process (or another Docker container) is already using that port.

**Fix:**
```bash
# Find what's using port 8000
sudo lsof -i :8000         # Linux/Mac
netstat -ano | findstr 8000  # Windows

# Kill the process
sudo kill -9 <PID>

# Or just change the host port in docker-compose.yml
ports:
  - "9000:8000"     # Changed left side to 9000
# Then visit localhost:9000
```

---

### Issue 7 — `pip install` fails for packages like FAISS, PyTorch, or `psycopg2`

**Cause:** Some packages require C/C++ compilers or system libraries that aren't present in slim Python images.

**Fix:**
```dockerfile
# Install build dependencies before pip install
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    cmake \
    libpq-dev \           # for psycopg2
    libopenblas-dev \     # for FAISS/numpy
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -r requirements.txt
```

---

### Issue 8 — `faiss-gpu` or GPU packages not found / `CUDA not available`

**Cause:** GPU packages need a CUDA-enabled base image.

**Fix:**
```dockerfile
# Replace python:3.11-slim with a CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*
```

```bash
# Run container with GPU access
docker run --gpus all -p 8000:8000 my-recommender:latest
```

---

### Issue 9 — FastAPI returns 422 Unprocessable Entity

**Cause:** This is a Pydantic validation error. The request body or query parameters don't match the expected schema.

**Fix:**
```bash
# Check the full error detail
curl -X POST http://localhost:8000/recommend \
     -H "Content-Type: application/json" \
     -d '{"query": "graph networks"}'

# The response body will have a 'detail' field explaining exactly what's wrong
# Example: {"detail": [{"loc": ["body", "top_k"], "msg": "field required", "type": "value_error.missing"}]}
```

---

### Issue 10 — Container starts but API returns 500 Internal Server Error

**Cause:** A runtime error inside your code (model not found, wrong path, import error).

**Fix:**
```bash
# Check logs immediately
docker-compose logs api

# Or drop into the container and run manually
docker exec -it recommender-api /bin/bash
python -c "from src.models.recommender import load_model; load_model()"
```

---

### Issue 11 — `WARNING: Running pip as the 'root' user`

**Cause:** Not a breaking error, but a security warning. The container runs as root by default.

**Fix (optional but good practice):**
```dockerfile
# At the end of Dockerfile, create a non-root user
RUN useradd -m -u 1000 appuser
USER appuser
```

---

### Issue 12 — Model file not found inside container

**Cause:** You're loading a model from a path that exists locally but wasn't copied into the image, or you're using a wrong relative path.

**Fix:**
```dockerfile
# Copy models explicitly in the Dockerfile
COPY models/ ./models/

# Or mount at runtime (better for large models)
docker run -v $(pwd)/models:/app/models my-recommender:latest
```

```python
# Use environment variables for paths (safer)
import os
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/recommender.pkl")
```

---

### Issue 13 — `docker-compose up` says service is unhealthy

**Cause:** The healthcheck command is failing. This often means the app hasn't started yet, is crashing on startup, or the health endpoint doesn't exist.

**Fix:**
```bash
# Check the container status
docker inspect recommender-api | grep -A 10 Health

# Look at the logs to find the startup error
docker-compose logs api

# Temporarily remove the healthcheck from docker-compose.yml to get the container running for debugging
```

---

### Issue 14 — `uvicorn` workers crash under load (timeout / worker timeout)

**Cause:** Default Uvicorn is a single process. Under load or slow endpoints, it can time out.

**Fix:**
```dockerfile
# Use Gunicorn as a process manager with multiple Uvicorn workers
CMD ["gunicorn", "src.api.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]
```

---

### Issue 15 — Changes to code not reflected after `docker-compose up`

**Cause:** You changed your source code but didn't rebuild the image. Docker Compose doesn't auto-detect code changes inside the image.

**Fix:**
```bash
# Rebuild the specific service
docker-compose up --build api

# Or, for development, mount your source code as a volume (live reload)
# In docker-compose.yml:
volumes:
  - ./src:/app/src   # Changes on your machine instantly reflect inside container
```

---

### Issue 16 — `ssl.SSLError` or certificate verification errors inside container

**Cause:** Slim images sometimes have outdated or missing SSL certificates.

**Fix:**
```dockerfile
RUN apt-get update && apt-get install -y ca-certificates \
    && update-ca-certificates
```

---

### Issue 17 — Services can't reach each other in docker-compose

**Cause:** Using `localhost` to refer to another container. `localhost` inside any container refers to that container itself, not the host or another container.

**Fix:**
```yaml
# Use service names as hostnames
# api container connecting to mlflow container:
MLFLOW_TRACKING_URI: "http://mlflow:5000"    # ✅ correct
# NOT "http://localhost:5000"                 # ❌ wrong
```

---

### Issue 18 — `dvc pull` fails inside Docker container

**Cause:** DVC remote credentials aren't available inside the container.

**Fix:**
```dockerfile
# Copy DVC config into the image
COPY .dvc/config .dvc/config

# Pass credentials as environment variables
ENV GDRIVE_CREDENTIALS_DATA='{"client_id": "...", ...}'
```

Or pre-download data before building:
```bash
# Pull data locally first, then Docker copies the data folder
dvc pull
docker-compose build
```

---

## ✅ New Project Checklist

Use this every time you start a new ML project:

```
[ ] git init + git remote add origin
[ ] python -m venv .venv && pip install -r requirements.txt
[ ] dvc init
[ ] Create .gitignore (exclude .venv, __pycache__, .env, data/, models/)
[ ] Create .dvcignore
[ ] touch src/__init__.py (and every subfolder)
[ ] dvc remote add -d myremote <storage-url>
[ ] Create dvc.yaml and params.yaml
[ ] Pin requirements: pip freeze > requirements.txt
[ ] Dockerfile with: 0.0.0.0 host, PYTHONPATH, dos2unix, multi-stage build
[ ] docker-compose.yml with healthchecks and volume mounts
[ ] Create .dockerignore (exclude .venv, .git, __pycache__, mlruns)
[ ] mlflow.set_tracking_uri() in train.py
[ ] git add . && git commit -m "chore: project scaffold"
[ ] git push && dvc push
```

---

> 📝 **This document is a living reference. As you encounter new issues or build new features, add them here with the symptom, root cause, and fix.**
