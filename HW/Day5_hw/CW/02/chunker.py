from dataclasses import dataclass
from typing import List

@dataclass
class Chunk:
    chunk_id: str
    text: str
    start: int
    end: int
    method: str          # "fixed" or "sliding"
    source: str          # filename/path

def _clean_text(s: str) -> str:
    # 簡單清理：把多餘空白壓縮，但保留換行可讀性
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in s.split("\n")]
    # 去掉連續空行
    out = []
    prev_empty = False
    for ln in lines:
        empty = (ln == "")
        if empty and prev_empty:
            continue
        out.append(ln)
        prev_empty = empty
    return "\n".join(out).strip()

def fixed_chunk(text: str, chunk_size: int = 500, overlap: int = 100, source: str = "text.txt") -> List[Chunk]:
    """
    固定切塊：每塊 chunk_size 字元，重疊 overlap。
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    text = _clean_text(text)
    chunks: List[Chunk] = []
    step = chunk_size - overlap
    i = 0
    idx = 0
    while i < len(text):
        j = min(i + chunk_size, len(text))
        seg = text[i:j].strip()
        if seg:
            chunks.append(Chunk(
                chunk_id=f"{source}::fixed::{idx}",
                text=seg,
                start=i,
                end=j,
                method="fixed",
                source=source
            ))
            idx += 1
        i += step
    return chunks

def sliding_window(text: str, window_size: int = 500, stride: int = 400, source: str = "text.txt") -> List[Chunk]:
    """
    滑動視窗切塊：window_size 視窗大小，stride 步長（越小重疊越多）。
    """
    if stride <= 0:
        raise ValueError("stride must be > 0")

    text = _clean_text(text)
    chunks: List[Chunk] = []
    i = 0
    idx = 0
    while i < len(text):
        j = min(i + window_size, len(text))
        seg = text[i:j].strip()
        if seg:
            chunks.append(Chunk(
                chunk_id=f"{source}::sliding::{idx}",
                text=seg,
                start=i,
                end=j,
                method="sliding",
                source=source
            ))
            idx += 1
        if j == len(text):
            break
        i += stride
    return chunks
