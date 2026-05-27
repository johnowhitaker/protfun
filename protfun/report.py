from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_neighbor_plot(path: Path, neighbors: list[dict], top_n: int = 20) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(neighbors).head(top_n).iloc[::-1]
    labels = frame["uniprot_id"] + " | " + frame["name"].str.slice(0, 42)
    fig_height = max(5, 0.32 * len(frame) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    ax.barh(labels, frame["cosine_similarity"], color="#356b63")
    ax.set_xlabel("Cosine similarity to miraculin ESMC mean embedding")
    ax.set_ylabel("")
    ax.set_xlim(max(0, frame["cosine_similarity"].min() - 0.03), 1.0)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Nearest candidates in scoped UniProt corpus")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_markdown_report(
    path: Path,
    seed: dict,
    neighbors: list[dict],
    plot_relpath: str,
    corpus_size: int,
    model: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    top_rows = []
    for row in neighbors[:15]:
        top_rows.append(
            "| {rank} | {accession} | {name} | {organism} | {length} | {score:.4f} |".format(
                rank=row["rank"],
                accession=row["accession"],
                name=row["name"].replace("|", "\\|"),
                organism=row["organism"].replace("|", "\\|"),
                length=row["length"],
                score=row["cosine_similarity"],
            )
        )
    path.write_text(
        "\n".join(
            [
                "# Miraculin Neighbor Scout",
                "",
                f"Seed: `{seed['accession']}` / `{seed['uniprot_id']}` / {seed['name']} from {seed['organism']}.",
                f"Embedding model: `{model}`. Candidate corpus size: {corpus_size}.",
                "",
                f"![Nearest neighbor plot]({plot_relpath})",
                "",
                "## Top Candidates",
                "",
                "| Rank | Accession | Name | Organism | Length | Cosine |",
                "|---:|---|---|---|---:|---:|",
                *top_rows,
                "",
                "## Notes",
                "",
                "This first pass ranks a scoped UniProt corpus, not the full ESM Atlas.",
                "The corpus intentionally mixes miraculin-like homologs with known taste-modifying or sweet proteins, plus plant Kunitz/trypsin-inhibitor-like proteins as a broader neighborhood.",
                "The next step is to inspect these hits and then expand the search source once Atlas-scale programmatic neighbors are available.",
                "",
            ]
        )
    )

