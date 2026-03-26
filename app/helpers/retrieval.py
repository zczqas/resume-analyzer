import json
import os
from pathlib import Path
from typing import Protocol, cast
from uuid import uuid4

import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DATASET_FILE = DATA_DIR / "job_requirements.jsonl"
DATASET_DIR = DATA_DIR / "job_requirements"
CHROMA_DIR = DATA_DIR / "chroma"
EMBEDDING_REGISTRY_FILE = DATA_DIR / "embedding_registry.json"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("JOB_COLLECTION_NAME", "job_requirements")


class CollectionLike(Protocol):
    name: str

    def count(self) -> int: ...

    def add(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, str | int | float | bool]],
        embeddings: object,
    ) -> None: ...

    def query(
        self, *, query_embeddings: list[list[float]], n_results: int, include: list[str]
    ) -> dict[str, object]: ...


class ClientLike(Protocol):
    def get_or_create_collection(self, *, name: str) -> CollectionLike: ...

    def delete_collection(self, *, name: str) -> None: ...


class EmbeddingArrayLike(Protocol):
    def tolist(self) -> list[list[float]]: ...


class EncoderModelLike(Protocol):
    def encode(
        self, texts: list[str], *, convert_to_numpy: bool, normalize_embeddings: bool
    ) -> EmbeddingArrayLike: ...


def _read_dataset(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []

    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(cast(dict[str, object], json.loads(line)))
    return rows


def _sanitize_collection_name(name: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in name.strip())
    clean = clean.strip("_")
    return clean or COLLECTION_NAME


def _dataset_path_for_collection(collection_name: str) -> Path:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    return DATASET_DIR / f"{collection_name}.jsonl"


def _load_embedding_registry() -> dict[str, str]:
    if not EMBEDDING_REGISTRY_FILE.exists():
        return {}
    try:
        content = EMBEDDING_REGISTRY_FILE.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        parsed = cast(object, json.loads(content))
        if not isinstance(parsed, dict):
            return {}
        mapping: dict[str, str] = {}
        parsed_dict = cast(dict[object, object], parsed)
        for key, value in parsed_dict.items():
            mapping[str(key)] = str(value)
        return mapping
    except Exception:
        return {}


def _save_embedding_registry(registry: dict[str, str]) -> None:
    EMBEDDING_REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ = EMBEDDING_REGISTRY_FILE.write_text(
        json.dumps(registry, ensure_ascii=True, indent=2), encoding="utf-8"
    )


def _register_embedding_id(
    collection_name: str, embedding_id: str | None = None
) -> str:
    registry = _load_embedding_registry()
    chosen_id = (embedding_id or "").strip() or str(uuid4())
    registry[chosen_id] = collection_name
    _save_embedding_registry(registry)
    return chosen_id


def resolve_collection_name(embedding_id: str) -> str | None:
    key = embedding_id.strip()
    if not key:
        return None
    return _load_embedding_registry().get(key)


def _parse_uploaded_dataset(file_bytes: bytes) -> list[dict[str, object]]:
    text = file_bytes.decode("utf-8").strip()
    if not text:
        return []

    if text.startswith("["):
        parsed = cast(object, json.loads(text))
        if not isinstance(parsed, list):
            raise ValueError("JSON file must contain a list of job requirement objects")
        rows: list[dict[str, object]] = []
        for item in cast(list[object], parsed):
            if isinstance(item, dict):
                rows.append(cast(dict[str, object], item))
        return rows

    rows = []
    for line in text.splitlines():
        clean_line = line.strip()
        if not clean_line:
            continue
        parsed_line = cast(object, json.loads(clean_line))
        if isinstance(parsed_line, dict):
            rows.append(cast(dict[str, object], parsed_line))
    return rows


def _write_dataset_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            _ = handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def _normalize_record_ids(
    rows: list[dict[str, object]], collection_name: str
) -> tuple[list[dict[str, object]], int]:
    normalized_rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    generated_count = 0

    for idx, row in enumerate(rows, start=1):
        item = dict(row)
        candidate = str(item.get("id", "")).strip()

        if not candidate:
            title = str(item.get("title", "")).strip()
            title_slug = "".join(
                ch.lower() if ch.isalnum() else "_" for ch in title
            ).strip("_")
            if not title_slug:
                title_slug = "requirement"
            candidate = f"{collection_name}_{title_slug}_{idx}"
            generated_count += 1

        unique_id = candidate
        suffix = 2
        while unique_id in seen_ids:
            unique_id = f"{candidate}_{suffix}"
            suffix += 1

        item["id"] = unique_id
        seen_ids.add(unique_id)
        normalized_rows.append(item)

    return normalized_rows, generated_count


class Embedder:
    def __init__(self, model_name: str) -> None:
        self.model: EncoderModelLike
        self.model = cast(
            EncoderModelLike, cast(object, SentenceTransformer(model_name))
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return vectors.tolist()


class JobVectorStore:
    def __init__(
        self, persist_dir: Path = CHROMA_DIR, collection_name: str = COLLECTION_NAME
    ) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedder: Embedder
        self.client: ClientLike
        self.collection: CollectionLike
        self.embedder = Embedder(EMBEDDING_MODEL)
        self.client = cast(
            ClientLike, cast(object, chromadb.PersistentClient(path=str(persist_dir)))
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def build_from_file(self, dataset_path: Path = DATASET_FILE) -> int:
        rows = _read_dataset(dataset_path)
        if not rows:
            return 0

        ids: list[str] = []
        docs: list[str] = []
        metadatas: list[dict[str, str]] = []

        for idx, row in enumerate(rows):
            title = str(row.get("title", "")).strip()
            text = str(row.get("text", "")).strip()
            skills = row.get("skills", [])

            if not text:
                continue

            doc_id = str(row.get("id") or f"job-{idx + 1}")
            ids.append(doc_id)
            docs.append(text)
            skills_text = ""
            if isinstance(skills, list):
                skill_items = cast(list[object], skills)
                skills_text = ", ".join(str(item) for item in skill_items)
            else:
                skills_text = str(skills)
            metadatas.append(
                {
                    "title": title,
                    "skills": skills_text,
                }
            )

        if not docs:
            return 0

        embeddings = self.embedder.embed(docs)
        self.collection.add(
            ids=ids,
            documents=docs,
            metadatas=cast(list[dict[str, str | int | float | bool]], metadatas),
            embeddings=cast(object, embeddings),
        )
        return len(docs)

    def reset_collection(self) -> None:
        self.client.delete_collection(name=self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name
        )

    def search(self, query: str, top_k: int = 3) -> list[dict[str, object]]:
        clean_query = query.strip()
        if not clean_query:
            return []

        query_embedding = self.embedder.embed([clean_query])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=max(1, top_k),
            include=["documents", "metadatas", "distances"],
        )

        documents_result = cast(list[list[str]] | None, result.get("documents"))
        metadatas_result = cast(
            list[list[dict[str, object]]] | None, result.get("metadatas")
        )
        distances_result = cast(list[list[float]] | None, result.get("distances"))

        documents = documents_result[0] if documents_result else []
        metadatas = metadatas_result[0] if metadatas_result else []
        distances = distances_result[0] if distances_result else []

        hits: list[dict[str, object]] = []
        for idx, doc_text in enumerate(documents):
            metadata = metadatas[idx] if idx < len(metadatas) else {}
            distance = distances[idx] if idx < len(distances) else None
            hits.append(
                {
                    "title": str((metadata or {}).get("title", "")),
                    "skills": str((metadata or {}).get("skills", "")),
                    "text": str(doc_text),
                    "distance": distance,
                }
            )
        return hits


_store_cache: dict[str, JobVectorStore] = {}


def get_job_store(collection_name: str = COLLECTION_NAME) -> JobVectorStore:
    normalized_name = _sanitize_collection_name(collection_name)
    if normalized_name in _store_cache:
        return _store_cache[normalized_name]

    store = JobVectorStore(collection_name=normalized_name)
    if store.collection.count() == 0:
        dataset_path = (
            DATASET_FILE
            if normalized_name == COLLECTION_NAME
            else _dataset_path_for_collection(normalized_name)
        )
        _ = store.build_from_file(dataset_path=dataset_path)
    _store_cache[normalized_name] = store
    return store


def load_job_requirements_from_bytes(
    file_bytes: bytes,
    embedding_name: str,
    embedding_id: str | None = None,
) -> dict[str, object]:
    collection_name = _sanitize_collection_name(embedding_name)
    rows = _parse_uploaded_dataset(file_bytes)
    if not rows:
        raise ValueError("Uploaded dataset has no valid records")

    rows, generated_ids_count = _normalize_record_ids(rows, collection_name)

    dataset_path = _dataset_path_for_collection(collection_name)
    _write_dataset_rows(dataset_path, rows)

    store = get_job_store(collection_name)
    store.reset_collection()
    indexed_count = store.build_from_file(dataset_path=dataset_path)

    assigned_embedding_id = _register_embedding_id(
        collection_name=collection_name, embedding_id=embedding_id
    )

    return {
        "embedding_name": collection_name,
        "embedding_id": assigned_embedding_id,
        "dataset_path": str(dataset_path),
        "records_received": len(rows),
        "records_indexed": indexed_count,
        "generated_ids_count": generated_ids_count,
    }
