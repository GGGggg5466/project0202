# 課堂作業 02（CW/02）
**主題：文字切塊（Chunking）、向量嵌入與向量資料庫（VDB）檢索比較**

---

## 一、作業說明
本作業目的在於實作不同文字切塊策略（Fixed Chunk 與 Sliding Window），將文字資料轉換為向量後嵌入至向量資料庫（Qdrant），並實際進行檢索以比較兩種切塊方式在召回結果上的差異。

---

## 二、資料來源
- `text.txt`  
  作為主要文本資料，內容為 Graph RAG 與傳統 RAG 技術介紹。
- `table/` 資料夾  
  包含以下四個檔案，用於示範表格與結構化資料的處理：
  - `table_txt.md`
  - `table_html.html`
  - `Prompt_table_v1.txt`
  - `Prompt_table_v2.txt`

---

## 三、文字切塊方法實作

### 1. 固定切塊（Fixed Chunk）
- 使用固定大小的文字區塊（chunk size = 500）
- 相鄰區塊保留重疊區域（overlap = 100）
- 優點：段落邊界明確、結構清楚  
- 缺點：跨段落資訊可能被切斷

### 2. 滑動視窗切塊（Sliding Window）
- 視窗大小（window size = 500）
- 步長（stride = 400），形成文字重疊
- 優點：保留較完整的上下文資訊，適合多跳推理
- 缺點：可能產生重複內容

---

## 四、向量嵌入（Embedding）
- 使用教師提供之 Embedding API：


## 📁 檔案與資料夾說明（File Description）

### Python 原始碼
- **`main.py`**  
  專案主程式入口。  
  負責整合整體流程，包括文字資料讀取、文字切塊、呼叫 Embedding API、將向量嵌入至 Qdrant 向量資料庫，以及執行檢索並輸出比較結果。

- **`chunker.py`**  
  文字切塊模組，實作兩種切塊策略：
  - 固定切塊（Fixed Chunk）
  - 滑動視窗切塊（Sliding Window）  
  用於比較不同切塊方式對向量檢索結果的影響。

- **`embed_client.py`**  
  Embedding API 呼叫模組。  
  封裝教師提供之嵌入服務，將文字區塊轉換為 4096 維向量表示，作為向量檢索的基礎。

- **`vdb_qdrant.py`**  
  向量資料庫操作模組。  
  使用 Qdrant 作為向量資料庫，負責：
  - Collection 建立
  - 向量資料寫入（upsert）
  - 向量相似度搜尋（query）

- **`table_loader.py`**  
  表格資料讀取與前處理模組。  
  將 Markdown、HTML 與 TXT 等不同格式的表格資料轉換為純文字後，納入向量化與檢索流程。

---

### 📤 輸出結果（outputs）
- **`outputs/chunks_fixed.jsonl`**  
  使用固定切塊方式所產生的文字區塊結果，每一行代表一個切塊，包含來源與位置資訊。

- **`outputs/chunks_sliding.jsonl`**  
  使用滑動視窗切塊方式所產生的文字區塊結果，用於與固定切塊方法進行比較。

- **`outputs/retrieval_compare.md`**  
  檢索實驗結果比較報告，呈現 Fixed Chunk 與 Sliding Window 在相同查詢條件下的召回結果差異。

---

### 📄 資料檔案
- **`text.txt`**  
  主要實驗文本資料來源，內容為 Graph RAG 與傳統 RAG 技術之介紹與比較。

- **`table/` 資料夾**  
  表格與結構化資料範例，用於示範不同格式資料的處理方式：
  - `table_txt.md`：Markdown 格式表格文字
  - `table_html.html`：HTML 格式表格資料
  - `Prompt_table_v1.txt`、`Prompt_table_v2.txt`：表格處理與提示設計範例

---

### ⚙️ 其他檔案
- **`requirements.txt`**  
  專案所需之 Python 套件清單。

- **`README.md`**  
  專案與作業說明文件，包含實作流程、方法說明與實驗結果。

- **`__pycache__/` 與 `*.pyc` 檔案**  
  Python 執行時自動產生之快取檔案，與作業實作內容無直接關聯，可不必上傳。