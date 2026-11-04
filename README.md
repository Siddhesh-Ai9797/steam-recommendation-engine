# Steam Games Recommendation Engine

[![CI](https://github.com/Siddhesh-Ai9797/steam-recommendation-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Siddhesh-Ai9797/steam-recommendation-engine/actions/workflows/ci.yml)

A multimodal recommendation system for Steam games, combining graph neural networks, NLP, and computer vision into a single fused model, trained with PyTorch DDP across multiple GPUs and served through a FAISS-backed FastAPI endpoint.

## Architecture

**Phase 1, Data and Graph Construction**
Built a bipartite user-game interaction graph from the Kaggle "Game Recommendations on Steam" dataset. Filtered to active users, 37,160 users and 36,229 games at dev scale, 213,364 users and 37,015 games at production scale, 8.47 million edges.

**Phase 2, GNN Candidate Generation**
Trained a SAGEConv-based heterogeneous graph neural network on link prediction with negative sampling. Achieved 0.9404 test AUC. Diagnosed moderate popularity bias in the candidate generator, expected at this stage, corrected in the fusion step.

**Phase 3, NLP and CV Fusion**
Fused three signals into one 128-dimensional item representation per game, the GNN embedding, a sentence-transformer embedding of the game's description, and a CLIP embedding of its header art. Trained the fusion layer with in-batch negative sampling after two earlier attempts collapsed under single-negative sampling. Final Recall@10 of 0.0112 on 20,000 held-out validation pairs, roughly 40x better than random.

**Phase 4, Distributed Training**
Retrained the GNN on the full production graph using PyTorch DDP across 2 Kaggle T4 GPUs. Achieved a 1.83x speedup over single-GPU training, 1069.52s down to 585.05s for 100 epochs.

**Phase 5, Serving**
Built a FAISS exact inner product index over the final fused game embeddings, wrapped in a FastAPI endpoint returning top-K recommendations for any user.

## Key Results

| Metric | Result |
|---|---|
| GNN candidate generator, test AUC | 0.9404 |
| Fusion model, Recall@10 | 0.0112, ~40x random baseline |
| DDP speedup, 1 GPU to 2 GPUs | 1.83x |
| Production graph scale | 213,364 users, 37,015 games, 8.47M edges |

## Stack

PyTorch, PyTorch Geometric, PyTorch DDP, sentence-transformers, CLIP, FAISS, FastAPI, Kaggle T4 x2 GPUs

## Project structure

```
src/steam_rec/      Library code: data, models, training loops, metrics, FAISS index, API
scripts/            Pipeline steps, run in order below
tests/              Unit and API tests on small synthetic data
notebooks/          Original Kaggle notebook with the full research and debugging log
```

## Reproducing the pipeline

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e ".[embed]"

python scripts/prepare_data.py --data-dir path/to/kaggle/steam
python scripts/train_gnn.py                                 # one GPU
torchrun --nproc_per_node=2 scripts/train_ddp.py            # or DDP across GPUs
python scripts/embed_content.py --data-dir path/to/kaggle/steam
python scripts/train_fusion.py                              # writes artifacts/
```

## Serving

```bash
uvicorn steam_rec.serve:app --reload
# or
docker build -t steam-rec . && docker run -p 8000:8000 steam-rec
```

| Endpoint | Returns |
|---|---|
| `GET /recommend/{user_node_id}?k=10` | Top K games for a user |
| `GET /similar/{app_id}?k=10` | Games closest to a given game |
| `GET /health` | Index size and status |

## Tests

```bash
pip install -e ".[test]"
pytest -q
```

## Notes

Full build log covering real debugging along the way, representation collapse in the fusion layer, a transformers library breaking change in CLIP's API, Windows CUDA DLL issues, and the fixes for each, is documented in the notebook itself.