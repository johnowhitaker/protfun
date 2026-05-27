from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import modal
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from protfun.modal_app import app, embed_records
from protfun.uniprot import best_uniparc_xref, get_by_accession


def parse_fasta(path: Path) -> list[dict]:
    records = []
    header = None
    sequence_parts: list[str] = []
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append(parse_header(header, "".join(sequence_parts)))
            header = line[1:]
            sequence_parts = []
        else:
            sequence_parts.append(line.strip())
    if header is not None:
        records.append(parse_header(header, "".join(sequence_parts)))
    return records


def parse_header(header: str, sequence: str) -> dict:
    parts = header.split("|")
    protein_hash = parts[0]
    source_id = parts[1] if len(parts) > 1 else ""
    source_db = parts[2] if len(parts) > 2 else ""
    return {
        "protein_hash": protein_hash,
        "source_id": source_id,
        "source_db": source_db,
        "accession": source_id or protein_hash,
        "uniprot_id": source_id or protein_hash,
        "name": "",
        "organism": "",
        "length": len(sequence),
        "sequence": sequence,
        "query": "atlas-export:miraculin-like-cluster",
    }


def sparse_vector(sparse: dict) -> dict[int, float]:
    indices = sparse.get("indices", [[]])[0]
    values = sparse.get("values", [])
    return {int(index): float(value) for index, value in zip(indices, values)}


def sparse_cosine(left: dict[int, float], right: dict[int, float]) -> float:
    shared = set(left) & set(right)
    dot = sum(left[index] * right[index] for index in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return float("nan")
    return dot / (left_norm * right_norm)


def top_features(vector: dict[int, float], n: int = 12) -> str:
    return ";".join(
        f"{index}:{value:.3g}"
        for index, value in sorted(vector.items(), key=lambda item: item[1], reverse=True)[:n]
    )


def parse_pdb_metrics(structures_dir: Path) -> dict[str, dict]:
    metrics = {}
    for path in structures_dir.glob("*.pdb"):
        protein_hash = path.stem
        row = {"pdb_path": str(path), "ptm": "", "mean_plddt": ""}
        for line in path.open():
            if not line.startswith("REMARK"):
                break
            if line.startswith("REMARK ptm"):
                row["ptm"] = line.split()[-1]
            elif line.startswith("REMARK mean_plddt"):
                row["mean_plddt"] = line.split()[-1]
        metrics[protein_hash] = row
    return metrics


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def load_existing_embeddings(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    embeddings = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "embedding" in row:
            embeddings[row["protein_hash"]] = row["embedding"]
    return embeddings


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cluster-dir", default="data/miraculin-like-cluster")
    parser.add_argument("--out", default="runs/miraculin/atlas_cluster_candidates.csv")
    parser.add_argument("--model", default="esmc-600m-2024-12")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cluster_dir = Path(args.cluster_dir)
    out_path = Path(args.out)
    fasta_records = parse_fasta(cluster_dir / "sequences.fasta")
    if args.limit:
        fasta_records = fasta_records[: args.limit]

    cluster_info = json.loads((cluster_dir / "cluster_info.json").read_text())
    cluster_rep = next(iter(cluster_info["clusters"]))
    cluster_meta = cluster_info["clusters"][cluster_rep]
    features = json.loads((cluster_dir / "features.json").read_text())
    pdb_metrics = parse_pdb_metrics(cluster_dir / "structures")

    feature_vectors = {
        protein_hash: sparse_vector(payload["protein_level"])
        for protein_hash, payload in features.items()
        if protein_hash in {record["protein_hash"] for record in fasta_records}
    }
    rep_vector = feature_vectors.get(cluster_rep)

    enriched = []
    for record in fasta_records:
        if record["source_db"] == "uniparc":
            try:
                record.update(best_uniparc_xref(record["source_id"]))
            except Exception as exc:
                record["xref_error"] = str(exc)
        record["name"] = record.get("protein_name") or record["source_id"]
        record["organism"] = record.get("organism", "")
        record["uniprot_id"] = record.get("uniprot_accession") or record["source_id"]
        record["accession"] = record.get("uniprot_accession") or record["source_id"]
        enriched.append(record)

    seed = get_by_accession("P13087").as_dict()
    embed_cache_path = out_path.with_suffix(".embeddings.jsonl")
    cached = load_existing_embeddings(embed_cache_path)
    missing = [record for record in enriched if record["protein_hash"] not in cached]
    seed_key = "__seed_P13087__"
    if seed_key not in cached:
        seed_record = {**seed, "protein_hash": seed_key}
        missing = [seed_record] + missing

    if missing:
        with modal.enable_output():
            with app.run():
                embedded_missing = embed_records.remote(missing, args.model)
        cached_rows = []
        if embed_cache_path.exists():
            cached_rows = [json.loads(line) for line in embed_cache_path.read_text().splitlines()]
        write_jsonl(embed_cache_path, cached_rows + embedded_missing)
        cached = load_existing_embeddings(embed_cache_path)

    seed_vec = np.array(cached[seed_key], dtype=np.float32)
    rows = []
    for record in enriched:
        protein_hash = record["protein_hash"]
        vec = np.array(cached[protein_hash], dtype=np.float32)
        sae_vec = feature_vectors.get(protein_hash, {})
        row = {
            "rank": 0,
            "primary_research_id": record.get("uniprot_accession")
            or record.get("xref_id")
            or record["source_id"]
            or protein_hash,
            "protein_hash": protein_hash,
            "source_db": record["source_db"],
            "source_id": record["source_id"],
            "uniparc_id": record.get("uniparc_id", ""),
            "uniprot_accession": record.get("uniprot_accession", ""),
            "uniprot_id_or_accession": record.get("uniprot_accession", ""),
            "xref_database": record.get("xref_database", ""),
            "xref_id": record.get("xref_id", ""),
            "xref_active": record.get("xref_active", ""),
            "gene_name": record.get("gene_name", ""),
            "protein_name": record.get("protein_name") or record.get("name", ""),
            "organism": record.get("organism", ""),
            "common_name": record.get("common_name", ""),
            "taxon_id": record.get("taxon_id", ""),
            "length": record["length"],
            "cosine_to_miraculin_esmc_mean": cosine(seed_vec, vec),
            "cluster_rep_hash": cluster_rep,
            "sae_cosine_to_cluster_rep": sparse_cosine(sae_vec, rep_vector) if rep_vector else "",
            "top_sae_features": top_features(sae_vec),
            "has_exported_structure": protein_hash in pdb_metrics,
            "pdb_path": pdb_metrics.get(protein_hash, {}).get("pdb_path", ""),
            "ptm": pdb_metrics.get(protein_hash, {}).get("ptm", ""),
            "mean_plddt": pdb_metrics.get(protein_hash, {}).get("mean_plddt", ""),
            "cluster_pct_characterized": cluster_meta.get("cluster_pct_characterized", ""),
            "cluster_mean_domain_coverage": cluster_meta.get("cluster_mean_domain_coverage", ""),
            "cluster_top_pfam_domains": ";".join(
                f"{pfam}:{payload.get('name', '')}:{payload.get('count', '')}"
                for pfam, payload in cluster_meta.get("cluster_top_pfam_domains", {}).items()
            ),
            "sequence": re.sub(r"[^A-Z]", "", record["sequence"]),
        }
        rows.append(row)

    rows.sort(key=lambda row: row["cosine_to_miraculin_esmc_mean"], reverse=True)
    for index, row in enumerate(rows, start=1):
        row["rank"] = index

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path} ({len(rows)} rows)")
    for row in rows[:12]:
        print(
            f"{row['rank']:>3}. {row['cosine_to_miraculin_esmc_mean']:.4f} "
            f"{row['uniprot_accession'] or row['source_id']:<14} "
            f"{row['protein_name'][:48]:<48} {row['organism']}"
        )


if __name__ == "__main__":
    main()
