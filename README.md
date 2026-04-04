# GraphSAGE Paper Recommender — Complete Final Plan

---

## Project Goal

A research paper recommendation system that finds papers related through **3-hop citation chain structure** — papers a researcher would never find by looking at direct citations or doing keyword search, but are genuinely relevant because they occupy similar positions in the citation network.

The core claim: lightweight text embeddings (MiniLM) enriched with 3-hop citation graph structure via GraphSAGE can match heavy citation-specialised models (SPECTER2) at nearly half the inference latency and a fraction of the model size.

---

## What the system does

Input: a paper title, abstract, or short description

Output: ranked list of papers that are related through citation neighbourhood structure — including papers 2-3 hops away that are invisible to direct citation search or keyword matching

The mechanism:
- MiniLM encodes each paper's text into a 384-dim vector capturing what the paper is about
- GraphSAGE enriches that vector with 3-hop citation neighbourhood context — absorbing signals from papers 1, 2, and 3 citation hops away
- The 3-hop discovery emerges automatically from neighbourhood aggregation — you never explicitly train on 3-hop pairs, the model generalises there on its own
- FAISS finds papers with similar enriched embeddings
- MMR removes redundancy from the final list ensuring diverse results

---

## What makes this different from keyword search and direct citation lookup

| Approach | What it finds |
|---|---|
| Keyword search | Papers using similar words |
| Direct citation lookup | Papers the query paper explicitly cites |
| SPECTER2 alone | Papers in the same direct citation cluster |
| **This system** | Papers sharing similar 3-hop citation neighbourhood structure, including non-obvious related papers never visible in direct citations |

---

## Datasets

### Cora — pipeline testing only

| Property | Value |
|---|---|
| Papers | 2,708 |
| Citation edges | 5,429 |
| Domain | Machine learning papers |
| Access | Built into PyTorch Geometric, zero manual download |
| PyG name | `torch_geometric.datasets.Planetoid(name='Cora')` |
| Size | ~3MB |
| Purpose | Test entire pipeline end to end before committing hours to OGBN-Arxiv |

### OGBN-Arxiv — main training and evaluation

| Property | Value |
|---|---|
| Papers | 169,343 |
| Citation edges | 1,166,243 |
| Domain | Computer Science (40 subfields) |
| Access | Auto-downloads via `ogb` package |
| Size | ~500MB |
| Splits | Pre-made temporal: train up to 2017, val 2018, test 2019 |
| Official page | https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv |
| Direct zip | https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip |

### Title and abstract file — manual download required

| Property | Value |
|---|---|
| URL | https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz |
| Size | ~90MB compressed |
| Contents | paper_id, title, abstract for all 169,343 papers |
| Required for | MiniLM encoding — without this file you cannot encode papers |
| Save to | `data/raw/titleabs.tsv.gz` |

---

## Models

### MiniLM — text encoding

| Property | Value |
|---|---|
| HuggingFace name | `sentence-transformers/all-MiniLM-L6-v2` |
| HuggingFace page | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 |
| Size | 22MB |
| Output dimension | 384 |
| Max input tokens | 256 (enough for title + abstract) |
| Encoding batch size | 64 |
| Encoding time 169K papers CPU | ~20 minutes |
| Query encoding latency | ~2ms |
| Download location | `~/.cache/huggingface/hub/` |
| Downloads automatically | Yes, on first encode run |

### GraphSAGE — trained by you

| Property | Value |
|---|---|
| What it is | Your own model trained from scratch |
| Input dimension | 384 (MiniLM output) |
| Hidden dimension | 256 |
| Output dimension | 128 |
| Layers | 3 |
| Aggregation | Mean |
| Normalisation | Layer norm after each layer |
| Dropout | 0.3 |
| Layer 1 neighbours | 10 |
| Layer 2 neighbours | 5 |
| Layer 3 neighbours | 5 |
| Trained on | OGBN-Arxiv citation graph |
| Saved to | `models/graphsage_best.pth` |

---

## Accounts required

| Account | Purpose | URL |
|---|---|---|
| GitHub | Code, CI/CD, GHCR container registry | github.com |
| DagsHub | Free hosted MLflow UI | dagshub.com |
| Hugging Face | Demo deployment | huggingface.co |
| Google | DVC remote via Google Drive | Any Gmail |

### DagsHub setup steps
- Create account
- Click Create Repository → Connect a GitHub Repository → select your repo
- Copy MLflow tracking URI from repo page (looks like `https://dagshub.com/yourusername/graphsage-recommender.mlflow`)
- Go to User Settings → Access Tokens → create a token
- Save: tracking URI, username, token — all go into `.env`

### Hugging Face setup steps
- Create account
- Settings → Access Tokens → New Token → Write permission
- Save token (shown only once)
- Create new Space when ready for demo: https://huggingface.co/new-space
- Settings: SDK = Gradio, Hardware = CPU Basic (free), Visibility = Public

### Google Drive setup steps
- Open Google Drive
- Create folder named `graphsage-dvc`
- Open it, copy folder ID from URL (long string after `/folders/`)
- Example: URL ends in `/folders/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74` → ID is `1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74`

---

## Software to install

| Software | URL | Purpose |
|---|---|---|
| Anaconda or Miniconda | anaconda.com/download | Python environment |
| Git | git-scm.com/downloads | Version control |
| Git LFS | git-lfs.github.com | Large file uploads to HF |
| Docker Desktop | docker.com/products/docker-desktop | Running containers |

Install in this order. Verify each with `--version` in terminal before moving on.

---

## Python packages — complete list

Create conda environment named `graphsage` with Python 3.11. Never install into base.

### Data and graph

| Package | Version | Purpose |
|---|---|---|
| ogb | latest | Downloads OGBN-Arxiv |
| torch | 2.1.0+cpu | Deep learning, CPU build only |
| torch-geometric | latest | GraphSAGE, Cora/OGBN loaders |
| torch-scatter | latest | PyG dependency |
| torch-sparse | latest | PyG dependency |
| numpy | latest | Array operations |
| pandas | latest | Metadata handling |
| scipy | latest | Sparse matrix operations |

Install PyTorch with `+cpu` flag. CUDA build is 3GB+. CPU build under 700MB.

### Embeddings and search

| Package | Version | Purpose |
|---|---|---|
| sentence-transformers | latest | Loads MiniLM, clean inference API |
| transformers | 4.36.0 | Required by sentence-transformers |
| tokenizers | latest | Required by transformers |
| faiss-cpu | 1.7.4 | HNSW vector search |
| scikit-learn | latest | Recall, NDCG, ILD metrics |

### API

| Package | Version | Purpose |
|---|---|---|
| fastapi | 0.104.0 | Two endpoint API |
| uvicorn[standard] | latest | Runs FastAPI |
| pydantic | 2.4.0 | Input output validation |
| python-dotenv | latest | Loads `.env` credentials |

### MLOps

| Package | Version | Purpose |
|---|---|---|
| mlflow | 2.8.0 | Experiment tracking, model registry |
| dagshub | latest | Connects MLflow to DagsHub |
| dvc | 3.30.0 | Data and pipeline versioning |
| dvc[gdrive] | 3.30.0 | Google Drive remote |

### Testing and quality

| Package | Version | Purpose |
|---|---|---|
| pytest | 7.4.0 | Test runner |
| pytest-cov | latest | Coverage reports |
| httpx | latest | FastAPI endpoint testing |
| ruff | latest | Linting and formatting |

### Demo and utilities

| Package | Version | Purpose |
|---|---|---|
| gradio | 4.0.0 | HF Spaces interface |
| tqdm | latest | Progress bars |
| pyyaml | latest | Reading params.yaml |
| loguru | latest | Structured logging |
| matplotlib | latest | Metric plots |

---

## Dataset files — where everything goes

```
data/
├── raw/
│   ├── titleabs.tsv.gz            ← MANUAL DOWNLOAD from SNAP URL above
│   └── ogbn_arxiv/                ← AUTO-CREATED by ogb package
│       ├── raw/
│       │   ├── data.npz           ← graph edges and node years
│       │   └── node-label.csv.gz  ← subject area labels
│       └── split/
│           └── time/
│               ├── train.csv.gz   ← train paper indices
│               ├── valid.csv.gz   ← validation paper indices
│               └── test.csv.gz    ← test paper indices
│
└── processed/
    ├── minilm_embeddings.pt       ← 169,343 × 384 tensor (~250MB)
    ├── citation_graph.pt          ← PyG Data object with features
    ├── hard_negatives.pkl         ← negative indices per training paper
    ├── faiss_index.bin            ← HNSW index (~100MB)
    └── papers_metadata.csv        ← paper_id, title, year, subject
```

The `ogbn_arxiv/` folder is created and populated automatically the first time your ingest script runs. You never touch it manually. The only file you manually download and place is `titleabs.tsv.gz`.

---

## Folder and file structure — complete

```
graphsage-recommender/
│
├── data/                              ← DVC tracked, fully local
│   ├── raw/
│   │   ├── titleabs.tsv.gz            ← manual download
│   │   └── ogbn_arxiv/                ← auto-created by ogb
│   └── processed/
│       ├── minilm_embeddings.pt
│       ├── citation_graph.pt
│       ├── hard_negatives.pkl
│       ├── faiss_index.bin
│       └── papers_metadata.csv
│
├── models/                            ← DVC tracked
│   └── graphsage_best.pth
│
├── src/
│   ├── data_prep/
│   │   ├── ingest.py                  ← handles Cora and OGBN-Arxiv
│   │   └── preprocess.py             ← splits, metadata CSV
│   ├── features/
│   │   ├── encode.py                 ← MiniLM batch encoding
│   │   └── negatives.py             ← hard negative mining
│   ├── model/
│   │   ├── graphsage.py             ← architecture definition
│   │   ├── train.py                 ← training loop + MLflow
│   │   └── evaluate.py             ← Recall@K, NDCG@K, ILD
│   ├── serving/
│   │   ├── app.py                   ← FastAPI two endpoints
│   │   └── recommender.py          ← FAISS search + MMR
│   └── pipelines/
│       └── retrain.py              ← manual retrain script
│
├── tests/
│   ├── test_model.py
│   └── test_api.py
│
├── docker-compose.yml
├── Dockerfile
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── .env                             ← never commit
├── .gitignore
├── .dockerignore
└── .github/
    └── workflows/
        └── ci.yml
```

---

## params.yaml — every value

```yaml
data:
  dataset: ogbn-arxiv
  num_papers: 169343
  num_edges: 1166243
  raw_titles_file: data/raw/titleabs.tsv.gz

features:
  model: sentence-transformers/all-MiniLM-L6-v2
  embedding_dim: 384
  encoding_batch_size: 64
  max_length: 256

model:
  hidden_dim: 256
  output_dim: 128
  num_layers: 3
  neighbor_samples: [10, 5, 5]
  dropout: 0.3
  aggregator: mean

training:
  temperature: 0.07
  num_hard_negatives: 15
  num_random_negatives: 5
  batch_size: 512
  epochs: 100
  learning_rate: 0.001
  early_stopping_patience: 10

evaluation:
  top_k: [5, 10, 20]
  ild_threshold: 0.50
  recall_tolerance: 0.02
  min_improvement_over_baseline: 0.03

inference:
  hnsw_m: 16
  hnsw_ef_construction: 100
  hnsw_ef_search: 50
  mmr_lambda: 0.7
  top_k_candidates: 100
  default_results: 10
  pseudo_neighbors: 5
```

---

## DVC pipeline stages

| Stage | Depends on | Produces |
|---|---|---|
| `ingest` | params.yaml | `data/raw/ogbn_arxiv/` |
| `preprocess` | `data/raw/` | `papers_metadata.csv`, graph structure |
| `encode` | `data/raw/titleabs.tsv.gz`, params | `minilm_embeddings.pt` |
| `build_graph` | embeddings + preprocessed graph | `citation_graph.pt` |
| `compute_baseline` | `minilm_embeddings.pt` | baseline Recall@10 logged to MLflow |
| `mine_negatives` | `minilm_embeddings.pt`, graph | `hard_negatives.pkl` |
| `train` | graph + negatives + params | `graphsage_best.pth` |
| `build_index` | `graphsage_best.pth` + graph | `faiss_index.bin` |

`dvc repro` runs all stages in order and skips any stage whose inputs have not changed.

---

## Training details

### InfoNCE loss setup

| Sample | Count | How selected |
|---|---|---|
| Anchor | 1 | Any training paper |
| Positive | 1 | A paper the anchor directly cites |
| Hard negatives | 15 | Top-50 MiniLM nearest neighbours of anchor, minus actual citations |
| Random negatives | 5 | Randomly drawn from full 169K corpus |

Hard negative mining: for each training paper, build a flat FAISS index from MiniLM embeddings, find its top-50 nearest neighbours, remove any that are actual citations in the edge_index, sample 15 from what remains. These are papers that look textually similar but have no citation relationship — exactly the hard cases that force the model to learn genuine structural differences.

### Experiments to run

| Run name | Change from default |
|---|---|
| `minilm-baseline` | MiniLM only, no GraphSAGE — FAISS flat index |
| `graphsage-default` | All default params |
| `graphsage-wider-sampling` | neighbor_samples: [15, 10, 5] |
| `graphsage-smaller-hidden` | hidden_dim: 128 |
| `graphsage-lower-temp` | temperature: 0.05 |
| `graphsage-higher-temp` | temperature: 0.10 |

All 6 runs visible on DagsHub. The `minilm-baseline` run is the most important — it is your ground truth benchmark. GraphSAGE must beat it by at least 3 percentage points on Recall@10.

### Latency comparison to log

| System | Query latency | Index size |
|---|---|---|
| MiniLM alone | ~7ms | ~250MB |
| MiniLM + GraphSAGE | ~18ms | ~100MB |
| SPECTER2 alone (reference) | ~18ms | ~200MB |

MiniLM + GraphSAGE matches SPECTER2 latency but with a much smaller index and a 20x smaller text encoder. This is the project's story.

---

## Evaluation metrics

### Baseline — first thing you compute, before any training

- Build flat FAISS index from raw MiniLM embeddings of test papers
- For each test paper retrieve top-10 nearest neighbours
- Compute fraction of actual cited papers appearing in top-10 — this is baseline Recall@10
- Log to MLflow as run `minilm-baseline`
- Write this number down — every GraphSAGE experiment is compared against it

### Three metrics tracked every 5 epochs

| Metric | What it measures | Target |
|---|---|---|
| Recall@10 | Fraction of cited papers in top 10 | Above 0.70 |
| NDCG@10 | Relevant papers ranked near top | Above 0.60 |
| ILD | Average pairwise distance between top-10 results | Above 0.50 |

---

## FastAPI — two endpoints only

### POST /recommend

Input:
- `query` — string, required
- `k` — integer, optional, default 10, max 50
- `lambda_param` — float, optional, default 0.7, range 0.0 to 1.0

Output: list of papers each with `paper_id`, `title`, `year`, `subject`, `score`

### GET /health

Input: nothing

Output: `status`, `model_version`, `papers_in_index`, `uptime_seconds`

### Query to response flow inside recommender.py

- Encode query with MiniLM → 384-dim vector
- Find 5 nearest papers in FAISS as pseudo-neighbours
- Run GraphSAGE forward pass with query vector + pseudo-neighbours → 128-dim query embedding
- FAISS search for top 100 nearest papers using 128-dim embedding
- MMR selects final K papers balancing relevance and diversity
- Join with `papers_metadata.csv` for titles, years, subjects
- Return

Total latency: ~18ms on CPU.

---

## Docker setup

### Two containers

| Container | Runs | Port |
|---|---|---|
| `app` | FastAPI + FAISS + model | 8000 |
| `mlflow` | MLflow tracking UI | 5000 |

### App container

- Base: `python:3.11-slim`
- Multi-stage: builder installs packages, runtime copies only what is needed
- Target size: under 1GB (MiniLM is 22MB, much lighter than SPECTER2)
- Data and model mounted as read-only volumes — not baked into image
- Restart container after retraining to pick up new weights — no rebuild needed

### Volumes mounted into app container

| Local path | Purpose |
|---|---|
| `./data/processed/` | FAISS index and metadata CSV |
| `./models/` | GraphSAGE checkpoint |

### .dockerignore

```
data/
models/
.git/
notebooks/
tests/
.env
*.pth
*.bin
```

---

## Retraining pipeline

Manual script only. No schedule. No automation. Run whenever you want.

```
python src/pipelines/retrain.py
```

### What it does

- Fetches current champion Recall@10 from MLflow registry
- Runs `dvc repro` — skips cached stages, rebuilds changed ones
- Trains new GraphSAGE model with current `params.yaml`
- Logs everything to MLflow on DagsHub
- Checks two conditions against champion:

| Condition | Requirement |
|---|---|
| Recall@10 | New ≥ champion minus 0.02 |
| ILD | New ≥ 0.50 |

- Both pass: overwrites `graphsage_best.pth`, rebuilds `faiss_index.bin`, registers new champion in MLflow
- Either fails: champion untouched, failed run logged to MLflow with metric diff

Every run — pass or fail — permanently logged to DagsHub. Full history always visible.

---

## GitHub Actions CI

One workflow, three jobs, triggers on every push.

| Job | What it does | Condition |
|---|---|---|
| `test` | pytest + coverage ≥80% + ruff lint | Every push |
| `build` | Docker build + push to GHCR tagged with commit SHA | Only if test passes |
| `notify` | Post summary with test results and image size | Always |

### GitHub Secrets to add

| Secret | Value |
|---|---|
| `DAGSHUB_USERNAME` | Your DagsHub username |
| `DAGSHUB_TOKEN` | Your DagsHub access token |
| `MLFLOW_TRACKING_URI` | URI from DagsHub |

`GITHUB_TOKEN` for GHCR is automatic — no secret needed.

---

## Hugging Face Spaces demo

### What you upload

| File | Size | Method |
|---|---|---|
| `graphsage_best.pth` | ~60MB | Git LFS |
| `faiss_index.bin` | ~100MB | Git LFS |
| `papers_metadata.csv` | ~30MB | Git LFS |
| `src/serving/` folder | ~10KB | Normal git |
| `app.py` (Gradio wrapper) | ~2KB | Normal git |
| `requirements.txt` (serving only) | <1KB | Normal git |

Total upload: ~190MB. Well under HF 1GB limit.

### Serving requirements.txt for HF

```
torch==2.1.0+cpu
torch-geometric
sentence-transformers
faiss-cpu==1.7.4
gradio==4.0.0
pydantic==2.4.0
pyyaml
loguru
pandas
numpy
scikit-learn
```

### Gradio interface

- Text input for query
- Slider for K (5 to 20, default 10)
- Slider for MMR lambda (0.3 to 1.0, default 0.7)
- Dataframe output: title, year, subject, score

---

## What downloads automatically and when

| What | Size | When |
|---|---|---|
| Cora dataset | ~3MB | First ingest run with Cora |
| OGBN-Arxiv dataset | ~500MB | First ingest run with OGBN-Arxiv |
| MiniLM weights | ~22MB | First encode run |
| MLflow Docker image | ~200MB | First `docker-compose up` |
| **Total** | **~725MB** | One time only |

`titleabs.tsv.gz` (~90MB) is the only manual download.

---

## Complete execution steps

---

### Phase 1 — Setup

- Create GitHub repo named `graphsage-recommender`, make it public
- Connect DagsHub to GitHub repo, copy MLflow tracking URI and access token
- Create Google Drive folder `graphsage-dvc`, copy folder ID
- Create Hugging Face account, create access token with Write permission
- Create conda environment `graphsage` with Python 3.11
- Install all packages listed above
- Create full folder structure
- Create `params.yaml` with all values above
- Create `requirements.txt`
- Create `.gitignore` — exclude `data/`, `models/`, `.env`, `__pycache__/`, `*.pyc`, `.dvc/cache`
- Create `.dockerignore` — exclude `data/`, `models/`, `.git/`, `tests/`, `.env`, `*.pth`, `*.bin`
- Create `.env` with DagsHub tracking URI, username, token — add to `.gitignore` immediately
- Run `git init`, `dvc init`
- Set Google Drive as DVC remote using folder ID
- Authenticate Google Drive by running a test `dvc push`
- Commit all config files to Git, push to GitHub
- Log a dummy MLflow experiment, verify it appears on DagsHub

---

### Phase 2 — Data pipeline

- Download `titleabs.tsv.gz` manually from the SNAP URL, save to `data/raw/`
- Write `src/data_prep/ingest.py` supporting both Cora and OGBN-Arxiv via params
- **Test on Cora first** — set dataset to Cora in params, run ingest, verify it works in minutes
- Switch params back to ogbn-arxiv
- Run ingest script — OGBN-Arxiv downloads automatically (~500MB)
- Verify: 169,343 nodes, 1,166,243 edges, temporal splits intact
- Write `src/data_prep/preprocess.py` — creates `papers_metadata.csv` with paper_id, title (from titleabs.tsv.gz), year, subject_area (from node labels)
- Add `data/raw/` and `data/processed/papers_metadata.csv` to DVC tracking
- Run `dvc push` to back up to Google Drive
- Commit `.dvc` files to Git

---

### Phase 3 — Feature engineering

- Write `src/features/encode.py`
- Load `sentence-transformers/all-MiniLM-L6-v2` — downloads 22MB automatically on first run
- Concatenate title and abstract for each paper with a separator
- Encode in batches of 64, show tqdm progress bar
- Save output tensor shape 169,343 × 384 to `data/processed/minilm_embeddings.pt`
- Takes ~20 minutes on CPU

- **Compute MiniLM baseline immediately after encoding** — critical step before any training
- Build flat FAISS IndexFlatIP from test set MiniLM embeddings, normalise vectors
- Retrieve top-10 for each test paper, compute Recall@10 and NDCG@10
- Log to MLflow as run `minilm-baseline`
- Write these numbers down — they are your permanent benchmark

- Write `src/data_prep/preprocess.py` graph building section
- Load OGBN-Arxiv graph structure from OGB
- Load MiniLM embeddings
- Create PyG `Data` object: `x` = 169,343 × 384 embeddings, `edge_index` = citation pairs, `y` = subject labels, train/val/test masks
- Save to `data/processed/citation_graph.pt`

- Write `src/features/negatives.py`
- For each training paper: find top-50 MiniLM nearest neighbours via FAISS flat index, remove actual citations from that list, sample 15 as hard negatives, note 5 random paper IDs
- Save to `data/processed/hard_negatives.pkl`

- Define all stages in `dvc.yaml`
- Run `dvc repro` — verify full pipeline runs cleanly
- Run `dvc push` to back up all processed files
- Commit `.dvc` files to Git

---

### Phase 4 — Model training

- Write `src/model/graphsage.py`
- 3-layer GraphSAGE using PyG `SAGEConv`
- 384 → 256 → 256 → 128 with layer norm and dropout after each layer
- Separate projection head 128 → 128 used only during training
- Mean aggregation at each layer

- Write `src/model/evaluate.py`
- Recall@K, NDCG@K using scikit-learn
- ILD: average pairwise cosine distance between top-K result embeddings

- Write `src/model/train.py`
- Read all params from `params.yaml` — nothing hardcoded
- Start MLflow run, log all params immediately
- Use PyG `NeighborLoader` with sampling [10, 5, 5]
- InfoNCE loss: 1 positive + 15 hard negatives + 5 random negatives, temperature 0.07
- Log train loss and val loss every epoch
- Log Recall@10, NDCG@10, ILD every 5 epochs to MLflow
- Save checkpoint when validation Recall@10 improves
- Early stopping at 10 epochs no improvement
- At end: evaluate on test set, log test metrics, register in MLflow Model Registry as `graphsage-challenger`

- Run all 6 experiments listed in the experiments table above
- After all 6 complete, open DagsHub, sort by test Recall@10
- Confirm best GraphSAGE run beats MiniLM baseline by at least 3 percentage points
- Register best as production champion in MLflow registry

- Write `build_index` pipeline stage in `src/` — generates 128-dim GraphSAGE embeddings for all 169K papers, builds FAISS HNSW (M=16, efConstruction=100), saves `faiss_index.bin`
- Verify: single FAISS query returns top-100 in under 10ms
- DVC-track `faiss_index.bin` and `graphsage_best.pth`
- Run `dvc push`

---

### Phase 5 — Serving

- Write `src/serving/recommender.py`
- `Recommender` class loads on startup: FAISS index, GraphSAGE weights, MiniLM model, metadata CSV
- `recommend` method: MiniLM encode → 5 pseudo-neighbours from FAISS → GraphSAGE forward pass → FAISS top-100 → MMR → metadata join → return list
- MMR: iteratively select paper maximising λ×(query similarity) − (1−λ)×(max similarity to already selected) until K papers chosen

- Write `src/serving/app.py`
- Single `Recommender` instance created at startup
- `POST /recommend`: Pydantic input model with `query`, `k` (default 10, max 50), `lambda_param` (default 0.7)
- Pydantic output model with list of papers each having `paper_id`, `title`, `year`, `subject`, `score`
- `GET /health`: returns status, model version, papers in index, uptime

- Write `tests/test_model.py`
- GraphSAGE forward pass produces correct output shape (batch × 128)
- InfoNCE loss decreases over a few iterations on dummy data
- FAISS index loads and returns correct number of results
- MMR output is more diverse than raw FAISS output measured by ILD

- Write `tests/test_api.py`
- `/health` returns 200 and contains all expected fields
- `/recommend` with valid query returns correct number of results
- `/recommend` with invalid input returns 422
- `/recommend` with k=1 returns exactly 1 result
- `/recommend` with k=20 returns exactly 20 results
- Coverage above 80% on `src/serving/`

---

### Phase 6 — Docker

- Write Dockerfile with two stages
- Stage 1 builder: `python:3.11-slim`, install all packages from requirements.txt
- Stage 2 runtime: fresh `python:3.11-slim`, copy installed packages from builder, copy `src/`, copy `params.yaml`, expose port 8000
- Do not copy `data/` or `models/` — mounted as volumes
- Target image size under 1GB (MiniLM is 22MB, much lighter than SPECTER2)

- Write `docker-compose.yml`
- `app` service: your Dockerfile, port 8000, mounts `./data/processed` and `./models` as read-only, loads `.env`
- `mlflow` service: official MLflow image, port 5000, mounts `./mlruns` folder

- Run `docker-compose up`
- Verify `localhost:8000/health` returns 200
- Verify `localhost:8000/docs` shows Swagger UI with both endpoints
- Test `/recommend` from Swagger UI with query "attention mechanism for sequences"
- Verify real paper titles appear in under 35ms
- Verify `localhost:5000` shows all training experiments from DagsHub

---

### Phase 7 — Retraining pipeline

- Write `src/pipelines/retrain.py`
- Fetch champion Recall@10 from MLflow registry at start
- Run `dvc repro` programmatically
- Train new model, log to MLflow
- Check evaluation gate: Recall@10 ≥ champion − 0.02 AND ILD ≥ 0.50
- Pass: overwrite `graphsage_best.pth`, rebuild `faiss_index.bin`, register new champion
- Fail: keep champion, log failure reason and metric diff to MLflow

- Test fail path: temporarily raise the threshold so new model cannot win, run script, confirm champion untouched and fail logged
- Test pass path: restore normal params, run script, confirm new champion registered and files updated
- After both paths verified, retraining pipeline is complete

---

### Phase 8 — CI/CD

- Create `.github/workflows/ci.yml`
- Three jobs: `test` → `build` → `notify`
- `test`: setup Python 3.11, install requirements, set DagsHub env vars from secrets, run pytest with coverage, run ruff
- `build`: login to GHCR with `GITHUB_TOKEN`, build image, push tagged with commit SHA and `latest`
- `notify`: post commit summary

- Add three GitHub secrets: `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`, `MLFLOW_TRACKING_URI`
- Push a change to GitHub, watch Actions tab
- Verify all three jobs go green
- Check GHCR in GitHub profile for the pushed image

---

### Phase 9 — HF Spaces demo

- Create `hf-demo/` folder separately from main project
- Copy into it: `src/serving/`, `params.yaml`, `graphsage_best.pth`, `faiss_index.bin`, `papers_metadata.csv`, serving `requirements.txt`
- Create root `app.py` with Gradio interface: text input, K slider, lambda slider, dataframe output
- Create HF Space at huggingface.co/new-space (Gradio, CPU Basic, Public)
- In `hf-demo/`: `git init`, `git lfs install`, track `.pth` `.bin` `.csv` with LFS
- Add HF remote URL, commit all files, push
- HF builds and deploys in 3-5 minutes
- Verify public URL returns real results

---

## Verification checklist

| Phase | What to verify |
|---|---|
| Phase 1 Setup | `dvc push` works, MLflow dummy run appears on DagsHub |
| Phase 2 Data | Cora pipeline runs, OGBN-Arxiv has 169,343 nodes and 1,166,243 edges |
| Phase 3 Features | `dvc repro` runs cleanly, MiniLM baseline Recall@10 written down and in MLflow |
| Phase 4 Training | 6 runs on DagsHub, best GraphSAGE beats baseline by ≥3%, model in MLflow registry |
| Phase 5 Serving | `/health` 200, `/recommend` returns results, tests ≥80% coverage |
| Phase 6 Docker | `docker-compose up` clean, Swagger UI works, results in under 35ms |
| Phase 7 Retrain | Both pass and fail paths work, history in DagsHub |
| Phase 8 CI | GitHub Actions green, image in GHCR |
| Phase 9 Demo | HF Spaces URL live and returning results |

---

## Minimum hardware

| Component | Minimum | Better |
|---|---|---|
| RAM | 8GB | 16GB |
| Free disk | 4GB | 8GB |
| CPU | Quad-core | 8-core |
| GPU | Not required | Speeds MiniLM encoding from 20min to 3min |

After initial setup everything runs fully offline. Internet only needed when pushing to GitHub, DagsHub, Google Drive, and HF Spaces.
>>>>>>> 86604c9 (Final GraphSAGE System - API, UI, Docker, CI/CD (Security Fix))
