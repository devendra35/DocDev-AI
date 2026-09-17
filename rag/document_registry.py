from __future__ import annotations

import hashlib
import json
from pathlib import Path


class DocumentRegistry:
    def __init__(
        self,
        registry_path: str | Path = "vectorstore/documents.json",
    ):
        self.registry_path = Path(registry_path)

        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.documents = self._load()

    def _load(self) -> dict:
        if not self.registry_path.exists():
            return {}

        try:
            with self.registry_path.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            if not isinstance(data, dict):
                return {}

            return data

        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self) -> None:
        temporary_path = self.registry_path.with_suffix(
            ".tmp"
        )

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.documents,
                file,
                indent=2,
                ensure_ascii=False,
            )

        temporary_path.replace(
            self.registry_path
        )

    @staticmethod
    def calculate_hash(
        file_path: str | Path,
    ) -> str:
        file_path = Path(file_path)

        sha256 = hashlib.sha256()

        with file_path.open("rb") as file:
            for block in iter(
                lambda: file.read(1024 * 1024),
                b"",
            ):
                sha256.update(block)

        return sha256.hexdigest()

    def contains_hash(
        self,
        file_hash: str,
    ) -> bool:
        return any(
            document.get("hash") == file_hash
            for document in self.documents.values()
        )

    def add(
        self,
        file_path: str | Path,
        file_hash: str,
        chunks: int,
        file_type: str,
    ) -> None:

        file_path = Path(file_path)

        self.documents[file_hash] = {
            "filename": file_path.name,
            "file_type": file_type,
            "hash": file_hash,
            "chunks": chunks,
        }

        self._save()

    def list_documents(self) -> list[dict]:
        return list(self.documents.values())