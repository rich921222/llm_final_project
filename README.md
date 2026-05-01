# 自然語言處理期末專題：課程講義檢索問答系統

本專題使用課程講義 PDF 作為資料集，將 PDF 文字抽出後，以 TF-IDF 進行相關頁面檢索，並提供一個簡單的 RAG 問答流程。

## 環境安裝

```powershell
pip install -r requirements.txt
```

## 1. 從 PDF 抽文字

將講義 PDF 放在 `data_pdf/` 後執行：

```powershell
python extract_pdf_text.py
```

會產生：

- `data_text/`：每份 PDF 對應的文字檔
- `data_text_pages.jsonl`：每頁一筆資料，包含 `source`、`page`、`text`

## 2. TF-IDF 檢索

```powershell
python tfidf_search.py "什麼是 Time-homogeneous Markov process？"
```

系統會將相鄰 3 頁視為一個檢索單位，計算 query 與每個 3-page window 的 TF-IDF cosine similarity，輸出前 3 名。

## 3. 問答系統

預設不需要 API key，使用本地抽取式回答：

```powershell
python rag_answer.py "Substitution Cipher 的總可能組合數大約是多少？"
```

也可以明確指定：

```powershell
python rag_answer.py "幾月幾號是期中考？" --llm extractive
```

## 4. 可選：使用 OpenAI API 生成自然語言回答

此模式不是必要功能。若沒有 API key，仍可使用預設的本地抽取式回答。

若要啟用 OpenAI 回答，請先設定環境變數。PowerShell 範例：

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

再執行：

```powershell
python rag_answer.py "什麼是馬可夫鏈？" --llm openai
```

也可以建立本機 `.env` 檔保存 key，但請勿上傳 `.env`：

```text
OPENAI_API_KEY=your_api_key_here
```

## 範例問題

```powershell
python rag_answer.py "什麼是 Time-homogeneous Markov process？"
python rag_answer.py "Substitution Cipher 的總可能組合數大約是多少？"
python rag_answer.py "幾月幾號是期中考？"
```

## 注意事項

- 請不要將 `.env`、API key 或任何私密金鑰上傳到 GitHub。
- 本系統預設不使用 OpenAI API，因此老師或助教沒有 API key 也能執行。
- `deep-translator` 用於可選的中文 query 翻譯；若翻譯失敗，系統仍會使用內建關鍵詞擴展。
