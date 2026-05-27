# Modal Setup

Protfun uses Modal for remote compute and keeps only small run artifacts locally.

## Secrets

Create these Modal secrets before running analyses:

```bash
modal secret create biohub-api-token BIOHUB_API_TOKEN=...
modal secret create huggingface-token HUGGINGFACE_HUB_TOKEN=... HF_TOKEN=...
```

The local `.env` file is ignored by git and can hold the same values for convenience.

## Smoke Test

```bash
modal run -m protfun.modal_app::app.smoke
```

This embeds UniProt `P13087` (miraculin) through the Biohub ESMC API and prints the embedding dimension.

## Miraculin Neighbor Demo

```bash
python scripts/miraculin_neighbors.py --per-query 30
```

The script builds a scoped UniProt corpus, embeds records remotely through Modal, ranks candidates by cosine similarity to miraculin, and writes a lightweight report under `runs/miraculin/`.

## Atlas Cluster Candidate Ranking

After exporting a cluster from the Biohub ESM Atlas UI into `data/miraculin-like-cluster.zip`, extract it and rank the contained sequences against canonical miraculin:

```bash
mkdir -p data/miraculin-like-cluster
unzip -o data/miraculin-like-cluster.zip -d data/miraculin-like-cluster
python scripts/rank_cluster_candidates.py
```

This writes `runs/miraculin/atlas_cluster_candidates.csv` with ESMC similarity, UniParc/UniProt cross-reference metadata, exported SAE feature summaries, and PDB confidence metrics when structures are present.
