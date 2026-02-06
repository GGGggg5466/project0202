import json
from pathlib import Path
from typing import List, Dict, Any

from chunker import fixed_chunk, sliding_window, Chunk
from embed_client import embed_texts
from table_loader import load_table_texts
from vdb_qdrant import QdrantVDB

OUTDIR = Path("outputs")
OUTDIR.mkdir(exist_ok=True)

COLLECTION = "cw02"
VECTOR_SIZE = 4096  # 你量到的 embedding 維度

def dump_jsonl(path: Path, chunks: List[Chunk]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "chunk_id": c.chunk_id,
                "source": c.source,
                "method": c.method,
                "start": c.start,
                "end": c.end,
                "text": c.text
            }, ensure_ascii=False) + "\n")

def build_points(chunks: List[Chunk], embeddings: List[List[float]]) -> List[Dict[str, Any]]:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length mismatch")

    points = []
    for i, (c, v) in enumerate(zip(chunks, embeddings)):
        points.append({
            "id": i,  # 這裡用整數 id 就好
            "vector": v,
            "payload": {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "method": c.method,
                "start": c.start,
                "end": c.end,
                "text": c.text
            }
        })
    return points

def make_compare_md(query: str, fixed_hits, sliding_hits) -> str:
    def fmt(hits):
        lines = []
        for r in hits:
            p = r["payload"]
            lines.append(
                f"- score={r['score']:.4f} | {p['chunk_id']}\n\n"
                f"```text\n{p['text']}\n```\n"
            )
        return "\n".join(lines)

    md = []
    md.append(f"# Retrieval Compare\n")
    md.append(f"## Query\n\n{query}\n")
    md.append("## Fixed Chunk Results\n")
    md.append(fmt(fixed_hits))
    md.append("\n## Sliding Window Results\n")
    md.append(fmt(sliding_hits))
    md.append("\n## Quick Notes (你可以照抄交作業)\n")
    md.append("- Fixed chunk：段落切得比較『整齊』，但遇到跨段資訊時可能被切斷。\n")
    md.append("- Sliding window：重疊帶來更高機率涵蓋完整語意，但可能出現較多重複內容。\n")
    return "\n".join(md)

def main():
    # 1) 讀 text.txt
    text_path = Path("text.txt")
    text = text_path.read_text(encoding="utf-8", errors="ignore")

    # 2) 產生 chunks：固定切塊、滑動視窗
    fixed_chunks = fixed_chunk(text, chunk_size=500, overlap=100, source="text.txt")
    sliding_chunks = sliding_window(text, window_size=500, stride=400, source="text.txt")

    dump_jsonl(OUTDIR / "chunks_fixed.jsonl", fixed_chunks)
    dump_jsonl(OUTDIR / "chunks_sliding.jsonl", sliding_chunks)

    # 3) table 資料夾：讀取並切塊（同樣做 fixed + sliding）
    table_texts = load_table_texts("table")
    table_fixed_all: List[Chunk] = []
    table_sliding_all: List[Chunk] = []
    for src, t in table_texts.items():
        table_fixed_all.extend(fixed_chunk(t, chunk_size=500, overlap=100, source=src))
        table_sliding_all.extend(sliding_window(t, window_size=500, stride=400, source=src))

    # 4) 全部 chunks 合併（一起塞進同一個 collection，用 payload.method 過濾比較）
    all_chunks = fixed_chunks + sliding_chunks + table_fixed_all + table_sliding_all

    # 5) Embedding（批次做，避免一次塞太多）
    BATCH = 32
    all_vectors: List[List[float]] = []
    for i in range(0, len(all_chunks), BATCH):
        batch_texts = [c.text for c in all_chunks[i:i+BATCH]]
        vecs = embed_texts(batch_texts, task_description="檢索技術文件", normalize=True)
        # 基本檢查維度
        if vecs and len(vecs[0]) != VECTOR_SIZE:
            raise ValueError(f"Embedding dim mismatch: got {len(vecs[0])}, expected {VECTOR_SIZE}")
        all_vectors.extend(vecs)

    # 6) 建 Qdrant collection + upsert
    vdb = QdrantVDB(collection=COLLECTION, vector_size=VECTOR_SIZE)
    vdb.recreate_collection()

    points = build_points(all_chunks, all_vectors)
    vdb.upsert_points(points)

    # 7) 做一次 retrieval 比較（固定 vs 滑動）
    query = "Graph RAG 相對於傳統 RAG 解決了哪些問題？請用三點概括。"
    qvec = embed_texts([query], task_description="檢索技術文件", normalize=True)[0]

    fixed_hits = vdb.search(qvec, top_k=5, method="fixed")
    sliding_hits = vdb.search(qvec, top_k=5, method="sliding")

    md = make_compare_md(query, fixed_hits, sliding_hits)
    (OUTDIR / "retrieval_compare.md").write_text(md, encoding="utf-8")

    print("✅ Done")
    print(f"- chunks_fixed: {OUTDIR/'chunks_fixed.jsonl'}")
    print(f"- chunks_sliding: {OUTDIR/'chunks_sliding.jsonl'}")
    print(f"- compare: {OUTDIR/'retrieval_compare.md'}")
    print(f"- Qdrant collection: {COLLECTION} (dim={VECTOR_SIZE})")

if __name__ == "__main__":
    main()
