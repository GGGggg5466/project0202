import requests
from typing import List

API_URL = "https://ws-04.wade0426.me/embed"

def embed_texts(texts: List[str], task_description: str = "檢索技術文件", normalize: bool = True) -> List[List[float]]:
    """
    Call teacher-provided embedding API.
    Returns: embeddings: List[vector], each vector is length 4096 (per your measurement).
    """
    resp = requests.post(
        API_URL,
        json={
            "texts": texts,
            "task_description": task_description,
            "normalize": normalize,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    if "embeddings" not in data:
        raise ValueError(f"Unexpected response keys: {list(data.keys())}")
    return data["embeddings"]
