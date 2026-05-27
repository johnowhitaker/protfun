from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable
from urllib.parse import urlencode

import requests


UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPARC_URL = "https://rest.uniprot.org/uniparc"


@dataclass(frozen=True)
class ProteinRecord:
    accession: str
    uniprot_id: str
    name: str
    organism: str
    length: int
    sequence: str
    query: str

    def as_dict(self) -> dict:
        return asdict(self)


def _protein_name(result: dict) -> str:
    description = result.get("proteinDescription", {})
    recommended = description.get("recommendedName", {})
    if recommended.get("fullName", {}).get("value"):
        return recommended["fullName"]["value"]
    submissions = description.get("submissionNames") or []
    if submissions and submissions[0].get("fullName", {}).get("value"):
        return submissions[0]["fullName"]["value"]
    alternatives = description.get("alternativeNames") or []
    if alternatives and alternatives[0].get("fullName", {}).get("value"):
        return alternatives[0]["fullName"]["value"]
    return result.get("uniProtkbId", "")


def search_uniprot(query: str, size: int = 50) -> list[ProteinRecord]:
    fields = [
        "accession",
        "id",
        "protein_name",
        "organism_name",
        "length",
        "sequence",
    ]
    params = {
        "query": query,
        "format": "json",
        "fields": ",".join(fields),
        "size": str(size),
    }
    url = f"{UNIPROT_SEARCH_URL}?{urlencode(params)}"
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    records: list[ProteinRecord] = []
    for result in response.json().get("results", []):
        sequence = (result.get("sequence") or {}).get("value")
        if not sequence:
            continue
        records.append(
            ProteinRecord(
                accession=result["primaryAccession"],
                uniprot_id=result.get("uniProtkbId", result["primaryAccession"]),
                name=_protein_name(result),
                organism=(result.get("organism") or {}).get("scientificName", ""),
                length=int((result.get("sequence") or {}).get("length", len(sequence))),
                sequence=sequence,
                query=query,
            )
        )
    return records


def build_miraculin_candidate_corpus(per_query: int = 50) -> list[ProteinRecord]:
    queries = [
        '"miraculin"',
        '"miraculin-like"',
        '"taste-modifying protein"',
        '"taste modifying protein"',
        '"sweet protein"',
        '"thaumatin"',
        '"curculin"',
        '"neoculin"',
        '"monellin"',
        '"brazzein"',
        '"mabinlin"',
        '"gurmarin"',
        'protein_name:"trypsin inhibitor" AND taxonomy_id:33090',
        'protein_name:"Kunitz" AND taxonomy_id:33090',
    ]
    deduped: dict[str, ProteinRecord] = {}
    for query in queries:
        for record in search_uniprot(query, size=per_query):
            deduped.setdefault(record.accession, record)
    return list(deduped.values())


def get_by_accession(accession: str) -> ProteinRecord:
    records = search_uniprot(f"accession:{accession}", size=1)
    if not records:
        raise ValueError(f"UniProt accession not found: {accession}")
    return records[0]


def records_to_dicts(records: Iterable[ProteinRecord]) -> list[dict]:
    return [record.as_dict() for record in records]


def fetch_uniparc(uniparc_id: str) -> dict:
    response = requests.get(f"{UNIPARC_URL}/{uniparc_id}.json", timeout=60)
    response.raise_for_status()
    return response.json()


def best_uniparc_xref(uniparc_id: str) -> dict:
    data = fetch_uniparc(uniparc_id)
    xrefs = data.get("uniParcCrossReferences") or []
    active_uniprot = [
        xref
        for xref in xrefs
        if xref.get("database", "").startswith("UniProtKB") and xref.get("active")
    ]
    inactive_uniprot = [
        xref
        for xref in xrefs
        if xref.get("database", "").startswith("UniProtKB") and not xref.get("active")
    ]
    other_active = [xref for xref in xrefs if xref.get("active")]
    xref = (active_uniprot or inactive_uniprot or other_active or xrefs or [{}])[0]
    organism = xref.get("organism") or {}
    return {
        "uniparc_id": uniparc_id,
        "xref_database": xref.get("database", ""),
        "uniprot_accession": xref.get("id", "")
        if xref.get("database", "").startswith("UniProtKB")
        else "",
        "xref_id": xref.get("id", ""),
        "xref_active": xref.get("active", ""),
        "gene_name": xref.get("geneName", ""),
        "protein_name": xref.get("proteinName", ""),
        "organism": organism.get("scientificName", ""),
        "common_name": organism.get("commonName", ""),
        "taxon_id": organism.get("taxonId", ""),
        "xref_count": len(xrefs),
    }
