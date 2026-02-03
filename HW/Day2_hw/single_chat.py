import json
import re
from openai import OpenAI

# ===== 你的 vLLM OpenAI-compatible endpoint =====
BASE_URL = "https://ws-02.wade0426.me/v1"
API_KEY = "vllm-token"

# 模型可以換：例如 "google/gemma-3-27b-it" 或老師指定那個
MODEL = "google/gemma-3-27b-it"


def extract_json_block(text: str) -> str:
    """
    從模型輸出中擷取 JSON（支援 ```json ... ``` 或純 {...}）
    回傳 JSON 字串（不含```）
    """
    if not text:
        raise ValueError("Empty response")

    # 1) 優先抓 ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    # 2) 退而求其次：抓第一個 {...}（最外層 JSON）
    m = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m:
        return m.group(1).strip()

    raise ValueError("No JSON object found in model output")


def json_extract(user_input: str) -> dict:
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

    system_prompt = """
你是一個資料提取助手。
請從使用者文字中提取以下資訊，並嚴格以 JSON 格式回傳（只能輸出 JSON，不要多餘文字）。
需要的欄位：name, phone, product, quantity, address
規則：
- phone 請保留原格式（例如 0912-345-678）
- quantity 請輸出數字（int）
- 如果缺少資訊，欄位值請填 null
""".strip()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.1,
        max_tokens=256,
    )

    raw = response.choices[0].message.content or ""
    json_str = extract_json_block(raw)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        # 解析失敗時，把原始輸出印出來最好除錯
        raise ValueError(
            "JSON 解析失敗。\n"
            f"模型原始輸出：\n{raw}\n\n"
            f"擷取到的 JSON：\n{json_str}\n\n"
            f"錯誤：{e}"
        )

    return data


def main():
    # 你可以改成 input() 互動
    user_input = "你好，我是陳大明，電話是 0912-345-678，我想要訂購 3 台筆記型電腦，下週五送到台中市北區。"

    result = json_extract(user_input)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
