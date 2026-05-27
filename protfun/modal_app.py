from __future__ import annotations

import json
from pathlib import Path

import modal


app = modal.App("protfun")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "esm@git+https://github.com/Biohub/esm.git@c94ed8d",
        "numpy",
        "requests",
        "torch",
    )
)


@app.function(
    image=image,
    secrets=[
        modal.Secret.from_name("biohub-api-token"),
        modal.Secret.from_name("huggingface-token"),
    ],
    timeout=60 * 60,
)
def embed_records(records: list[dict], model_name: str = "esmc-600m-2024-12") -> list[dict]:
    import os

    import numpy as np
    from esm.sdk import esmc_client
    from esm.sdk.api import ESMProtein, ESMProteinError, LogitsConfig

    token = os.environ["BIOHUB_API_TOKEN"]
    client = esmc_client(model=model_name, url="https://biohub.ai", token=token)
    embedded = []
    for record in records:
        protein = ESMProtein(sequence=record["sequence"])
        tensor = client.encode(protein)
        if isinstance(tensor, ESMProteinError):
            embedded.append({**record, "error": tensor.error_msg})
            continue
        output = client.logits(
            tensor,
            LogitsConfig(return_mean_embedding=True),
            return_bytes=True,
        )
        if isinstance(output, ESMProteinError):
            embedded.append({**record, "error": output.error_msg})
            continue
        embedding = output.mean_embedding
        if embedding is None:
            embedded.append({**record, "error": "Biohub returned no mean embedding"})
            continue
        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu()
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        embedded.append({**record, "embedding": vector.tolist()})
    return embedded


@app.local_entrypoint()
def smoke() -> None:
    from protfun.uniprot import get_by_accession

    seed = get_by_accession("P13087").as_dict()
    result = embed_records.remote([seed])
    print(json.dumps({k: v for k, v in result[0].items() if k != "embedding"}, indent=2))
    print("embedding_dim", len(result[0].get("embedding", [])))
