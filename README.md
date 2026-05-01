# 自然語言處理期末專題：課程講義檢索問答系統

本專題使用課程講義 PDF 作為資料集，將 PDF 文字抽出後，以 TF-IDF 進行相關頁面檢索，並提供簡單的 RAG 問答流程。

系統預設不需要 OpenAI API key，老師或助教可以直接使用本地抽取式回答。OpenAI API 只作為可選的自然語言生成模式。

## 環境安裝

```powershell
pip install -r requirements.txt
```

## 資料處理

將講義 PDF 放在 `data_pdf/` 後執行：

```powershell
python extract_pdf_text.py
```

會產生：

- `data_text/`：每份 PDF 對應的文字檔
- `data_text_pages.jsonl`：每頁一筆資料，包含 `source`、`page`、`text`、`original_text`

程式會偵測疑似 PDF 抽取亂碼的頁面。如果頁面被判定為亂碼，會保留 `original_text` 方便檢查，但不會把亂碼放進 `text` 參與檢索。

## 可選：翻譯講義中的中文

若要將含中文的講義頁面翻成英文，再把「原文 + 英文翻譯」一起存入 JSONL，可執行：

```powershell
python extract_pdf_text.py --translate-chinese
```

翻譯模式會額外產生：

- `translated_text`：Google Translate 產生的英文翻譯，只有含中文且未被判定為亂碼的頁面才會出現
- `translation_skipped`：若該頁疑似亂碼，會標記 `garbled_text_detected`

翻譯結果會暫存在 `translation_cache.json`，避免重複呼叫 Google Translate。此檔案已被 `.gitignore` 排除，不需要上傳 GitHub。

## TF-IDF 檢索

```powershell
python tfidf_search.py "什麼是 Time-homogeneous Markov process？"
```

系統會將相鄰 3 頁視為一個檢索單位，計算 query 與每個 3-page window 的 TF-IDF cosine similarity，並輸出前 3 名。

可顯示實際查詢用的擴展 query：

```powershell
python tfidf_search.py "tfidf計算方法" --show-query
```

目前針對中英文混合 query 做了以下處理：

- 中文 query 可透過 `deep-translator` 翻成英文
- 內建課程關鍵詞擴展，例如「老師」、「教授」、「期中考」、「tfidf」、「計算方法」
- 將 `tfidf`、`TFIDF`、`TF-IDF` 統一正規化成 `tf idf`
- 過濾常見中文停用字，減少短問句被錯誤頁面帶偏

## 問答系統

預設使用本地抽取式回答，不需要 API key：

```powershell
python rag_answer.py "Substitution Cipher 的總可能組合數大約是多少？"
```

也可以明確指定：

```powershell
python rag_answer.py "幾月幾號是期中考？" --llm extractive
```

課程資訊類問題會自動將 `c0_course_introduction.pdf` 前 6 頁加入 context，例如：

```powershell
python rag_answer.py "教這門課的人是誰"
python rag_answer.py "老師是誰"
python rag_answer.py "期中考是幾月幾號？"
```

這可以避免「老師 / 教授 / 教這門課的人」等不同說法找不到課程介紹頁。

## 可選：使用 OpenAI API 生成自然語言回答

此模式不是必要功能。若沒有 API key，仍可使用預設的本地抽取式回答。

PowerShell 設定方式：

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
```

使用 OpenAI 模式：

```powershell
python rag_answer.py "什麼是馬可夫鏈？" --llm openai
```

預設情況下，OpenAI 模式只能根據講義 context 回答。如果講義沒有答案，會說明講義中找不到。

若希望講義找不到答案時，GPT 可以補充一般知識，請加上：

```powershell
python rag_answer.py "問題" --llm openai --allow-general-answer
```

此模式會要求 GPT 將一般知識標註為：

```text
一般知識補充（非講義來源）
```

## 常用範例

```powershell
python rag_answer.py "什麼是 Time-homogeneous Markov process？"
python rag_answer.py "Substitution Cipher 的總可能組合數大約是多少？"
python rag_answer.py "幾月幾號是期中考？"
python rag_answer.py "tfidf計算方法"
python rag_answer.py "教這門課的人是誰" --llm openai
```

## 注意事項

- 不要將 `.env`、API key 或任何私密金鑰上傳到 GitHub。
- 本系統預設不使用 OpenAI API，因此沒有 API key 也能執行。
- `deep-translator` 需要網路；若翻譯失敗，系統仍會使用內建關鍵詞擴展。
- OpenAI API 模式會消耗 API credits；若要控制成本，請關閉 auto recharge。
