# Protfun

Protfun, messing with protein models.

This repo is a small workspace for exploring protein-model neighborhoods, starting
with Biohub's ESM tooling as a reference submodule in `esm/`. The goal is to make
future protein-verse explorations repeatable: keep heavyweight compute remote,
cache only lightweight results locally, and leave enough documentation that a new
question can reuse the same path.

## First Exploration: Miraculin

The first pass zoomed in on miraculin, the taste-modifying protein from miracle
fruit, and used ESMC embedding similarity to look for nearby plant proteins that
might share the same broad beta-trefoil/Kunitz-like architecture.

What we built:

- Modal-backed ESMC embedding calls through the Biohub API.
- UniProt/UniParc helpers for collecting and enriching candidate sequences.
- A scoped UniProt neighbor scout for known sweet/taste-modifying proteins and
  nearby Kunitz-like plant proteins.
- An Atlas export ranking script that takes a Biohub UI cluster download and
  writes an agent-ready candidate CSV.

The early miraculin-like candidate set is mostly unvalidated biology: strong
sequence/feature similarity, sparse functional annotation, and no clear reports
yet of miracle-fruit-like effects in the edible plants we checked. That makes it
useful as a discovery queue rather than a claim of activity.

## Layout

- `esm/`: Biohub ESM repo as a submodule for reference.
- `protfun/`: reusable Python helpers and Modal app code.
- `scripts/`: runnable exploration scripts.
- `docs/modal.md`: Modal setup and run notes.
- `data/`: ignored local input exports from tools like the ESM Atlas UI.
- `runs/`: ignored local outputs such as reports, plots, and candidate CSVs.

## Reproducing The Miraculin Runs

Set up the Modal/Biohub/Hugging Face secrets as described in `docs/modal.md`,
then run:

```bash
python scripts/miraculin_neighbors.py --per-query 30
python scripts/rank_cluster_candidates.py
```

The current local outputs live under `runs/miraculin/` and are intentionally not
committed.
