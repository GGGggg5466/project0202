from pathlib import Path
from typing import Dict
from bs4 import BeautifulSoup

def load_table_texts(table_dir: str = "table") -> Dict[str, str]:
    """
    讀取 table 資料夾四個檔案：
    - table_txt.md : 直接當文字
    - table_html.html : HTML 轉純文字
    - Prompt_table_v1.txt / v2.txt : 當作文字（也可當 prompt 說明文件）
    """
    d = Path(table_dir)
    out: Dict[str, str] = {}

    # 1) md
    md = d / "table_txt.md"
    if md.exists():
        out[str(md)] = md.read_text(encoding="utf-8", errors="ignore")

    # 2) html
    html = d / "table_html.html"
    if html.exists():
        raw = html.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(raw, "lxml")
        txt = soup.get_text("\n")
        out[str(html)] = txt

    # 3) prompts
    for name in ["Prompt_table_v1.txt", "Prompt_table_v2.txt"]:
        p = d / name
        if p.exists():
            out[str(p)] = p.read_text(encoding="utf-8", errors="ignore")

    return out
