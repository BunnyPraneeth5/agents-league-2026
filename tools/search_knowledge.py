"""Azure AI Search helper for grounded certification knowledge retrieval."""
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

try:
    from azure.core.credentials import AzureKeyCredential
    from azure.search.documents import SearchClient
except Exception:  # pragma: no cover - optional runtime dependency
    AzureKeyCredential = None
    SearchClient = None


def _search_sync(query: str) -> List[Dict[str, Any]]:
    endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    api_key = os.environ.get("AZURE_SEARCH_KEY")
    index_name = os.environ.get("AZURE_SEARCH_INDEX_NAME")

    if not endpoint or not api_key or not index_name or SearchClient is None or AzureKeyCredential is None:
        return []

    client = SearchClient(
        endpoint=endpoint,
        index_name=index_name,
        credential=AzureKeyCredential(api_key),
    )

    results = client.search(search_text=query, top=3)
    grounded_results = []

    for result in results:
        document = dict(result)
        grounded_results.append(
            {
                "content": document.get("snippet", ""),
                "source": document.get("metadata_storage_path", ""),
                "score": document.get("@search.score", 0),
            }
        )

    return grounded_results


async def search_knowledge(query: str) -> List[Dict[str, Any]]:
    """Search the configured Azure AI Search index and return the top 3 hits."""
    try:
        return await asyncio.to_thread(_search_sync, query)
    except Exception:
        return []
