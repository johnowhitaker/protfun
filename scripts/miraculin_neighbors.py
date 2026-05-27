from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import modal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from protfun.modal_app import app, embed_records
from protfun.report import write_jsonl, write_markdown_report, write_neighbor_plot
from protfun.uniprot import build_miraculin_candidate_corpus, get_by_accession, records_to_dicts


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/miraculin")
    parser.add_argument("--per-query", type=int, default=30)
    parser.add_argument("--model", default="esmc-600m-2024-12")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = get_by_accession("P13087")
    records = [seed]
    records.extend(build_miraculin_candidate_corpus(per_query=args.per_query))
    deduped = {record.accession: record for record in records}
    records = list(deduped.values())
    record_dicts = records_to_dicts(records)

    with modal.enable_output():
        with app.run():
            embedded = embed_records.remote(record_dicts, args.model)
    write_jsonl(out_dir / "embedded.jsonl", embedded)

    errors = [row for row in embedded if "error" in row]
    usable = [row for row in embedded if "embedding" in row]
    seed_row = next(row for row in usable if row["accession"] == "P13087")
    seed_vec = np.array(seed_row["embedding"], dtype=np.float32)

    neighbors = []
    for row in usable:
        if row["accession"] == "P13087":
            continue
        score = cosine(seed_vec, np.array(row["embedding"], dtype=np.float32))
        clean = {k: v for k, v in row.items() if k != "embedding"}
        clean["cosine_similarity"] = score
        neighbors.append(clean)
    neighbors.sort(key=lambda row: row["cosine_similarity"], reverse=True)
    for index, row in enumerate(neighbors, start=1):
        row["rank"] = index

    write_jsonl(out_dir / "neighbors.jsonl", neighbors)
    (out_dir / "errors.json").write_text(json.dumps(errors, indent=2, sort_keys=True))
    write_neighbor_plot(out_dir / "neighbors.png", neighbors)
    write_markdown_report(
        out_dir / "report.md",
        {k: v for k, v in seed_row.items() if k != "embedding"},
        neighbors,
        "neighbors.png",
        corpus_size=len(records),
        model=args.model,
    )

    print(f"Wrote {out_dir / 'report.md'}")
    print(f"Embedded {len(usable)} records; {len(errors)} errors")
    print("Top 10:")
    for row in neighbors[:10]:
        print(
            f"{row['rank']:>2}. {row['accession']:>10} {row['uniprot_id']:<18} "
            f"{row['cosine_similarity']:.4f} {row['name']} [{row['organism']}]"
        )


if __name__ == "__main__":
    main()
