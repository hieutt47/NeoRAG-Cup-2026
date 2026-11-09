# NeoRAG Cup 2026 — Evidence-based Vietnamese RAG

Hệ thống Retrieval-Augmented Generation tiếng Việt cho cộng đồng S2B: ProPTIT, Multimedia PTIT (MPC) và Game Creators Club (GCC). Dự án phát triển trên starter kit của Ban tổ chức NeoRAG Cup 2026, bổ sung hybrid retrieval, reranker đa ngôn ngữ, tìm kiếm bổ sung và đối chiếu bằng chứng trước khi sinh câu trả lời.

> **Pipeline được chọn:** `neorag.pipelines.evidence:EvidenceFeedbackPipeline`.
> **Kết quả Public được ghi nhận:** 87,43/100, theo kết quả hệ thống chấm do người dự thi cung cấp. Các biến thể thử nghiệm không mặc nhiên là bản tốt hơn và không được chọn thay thế nếu chưa vượt điều kiện đánh giá.

## 1. Kết quả và phạm vi công bố

| Chỉ số | Train — chấm cục bộ bằng bộ chấm nguyên bản | Public — kết quả được ghi nhận |
|---|---:|---:|
| Retrieval score | 97,68 | 92,18 |
| Generation score | 83,76 | 82,67 |
| Tổng điểm | **90,72** | **87,43** |
| String Presence | 94,49 | 97,86 |
| ROUGE-L | 51,67 | 51,61 |
| Groundedness | 98,94 | 98,91 |
| Response Relevancy | 69,33 | 59,94 |
| Noise Sensitivity — thấp hơn tốt hơn | 11,89 | 12,29 |
| Abstention | 100,00 | 100,00 |

Các metric trong bảng được biểu diễn trên thang 0–100. Điểm train là kết quả phát triển trên dữ liệu đã được phân tích nhiều lần, **không phải đánh giá trên holdout độc lập** và không đại diện cho điểm Private Test. Bảng không công bố kết quả Private Test.

Predictions, ZIP bài nộp, cache gateway và báo cáo từng lượt chạy được giữ cục bộ, không nằm trong Git. Vì vậy, bảng trên là bản ghi kết quả, không phải các báo cáo có thể kiểm chứng chỉ từ nội dung repository. Chạy lại cần key hợp lệ, trọng số reranker và có thể cho kết quả sinh khác khi không có cache gốc.

## 2. Kiến trúc bản được chọn

```text
Câu hỏi
  → Hybrid search: embedding bge-m3 + BM25
  → Rerank bằng BGE đa ngôn ngữ
  → Haiku đề xuất truy vấn bổ sung dựa trên nguồn tìm lần đầu
  → Tìm thêm nguồn, hợp nhất và xếp hạng theo độ phủ
  → Haiku lập bảng bằng chứng theo từng ý
  → Kiểm tra ID và trích dẫn nguyên văn bằng mã Python
  → Sắp xếp nguồn theo bằng chứng
  → Haiku trả lời từ tối đa 10 nguồn đầy đủ + bảng bằng chứng
  → Answer + danh sách ID corpus gốc
```

- **Hybrid retrieval:** kết hợp dense similarity và BM25 sau chuẩn hóa, trọng số 50/50.
- **Reranking:** điểm BGE và hybrid được phối hợp với trọng số mặc định 75/25.
- **Feedback retrieval:** đọc 5 nguồn đầu để đề xuất tối đa 3 truy vấn bổ sung; mỗi nhánh lấy 8 ứng viên.
- **Evidence ledger:** xác định ý cần trả lời, đối tượng, thời điểm và trích dẫn từ nguồn. Python chỉ chấp nhận trích dẫn xuất hiện nguyên văn trong tài liệu có ID hợp lệ.
- **Generation:** dùng `claude-haiku-5-5` qua gateway tương thích OpenAI của BTC; không truyền `temperature`.

**Giới hạn quan trọng:** trích dẫn tồn tại không đồng nghĩa với khẳng định được nguồn chứng minh về mặt ngữ nghĩa. Hệ thống vẫn có thể chọn ý thừa, nhầm phạm vi hoặc bỏ điều kiện. Response Relevancy và độ gọn của câu trả lời hiện là những điểm cần cải thiện; retrieval cao không bảo đảm câu trả lời tối ưu.

## 3. Cài đặt

### Yêu cầu

- Python 3.10+; chạy các lệnh từ thư mục gốc repository.
- Key gateway BTC nếu cần embedding hoặc sinh câu trả lời thật.
- Trọng số BGE ONNX theo bước dưới để chạy pipeline được chọn.
- Không cần GPU cho BGE ONNX INT8; phiên bản hiện tại dùng CPUExecutionProvider.

### Môi trường Python

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[app,dev]" onnxruntime tokenizers
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[app,dev]' onnxruntime tokenizers
```

Sao chép `.env.example` thành `.env`, sau đó điền `OPENAI_API_KEY` trên máy cá nhân. Không đưa key vào mã nguồn, README, notebook hoặc commit. Các biến endpoint/model tham khảo trong `.env.example` và [hướng dẫn gateway](docs/gateway.md).

### Tải reranker BGE

Trọng số không được đưa vào Git. Dùng snapshot cố định của repository `onnx-community/bge-reranker-v2-m3-ONNX`:

- Revision: `6f5ff65298512715a1e669753bc754d2bc8f367b`.
- Model: `onnx/model_quantized.onnx` → `.cache/bge-reranker-v2-m3/model_quantized.onnx`.
- Tokenizer: `tokenizer.json` → `.cache/bge-reranker-v2-m3/tokenizer.json`.

Lệnh Python dưới đây dùng được sau khi kích hoạt môi trường trên cả Windows và Unix. Nó chỉ tải hai file cần thiết, kiểm tra SHA256 và không ghi đè file đã tồn tại:

```bash
python -c "import pathlib, urllib.request, hashlib; root=pathlib.Path('.cache/bge-reranker-v2-m3'); root.mkdir(parents=True, exist_ok=True); base='https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX/resolve/6f5ff65298512715a1e669753bc754d2bc8f367b/'; files=[('onnx/model_quantized.onnx','model_quantized.onnx','912fc1215c2dbff6499700534bd8d31253af01573861abbfc43afd1fab6cce5d'),('tokenizer.json','tokenizer.json','8bf8afbfd11306bd872018c53bfdf2e160a56f8edbcf49933324404791c148d3')]; exec('for remote,name,expected in files:\n path=root/name\n if not path.exists(): urllib.request.urlretrieve(base+remote,path)\n digest=hashlib.file_digest(path.open(\"rb\"),\"sha256\").hexdigest() if hasattr(hashlib,\"file_digest\") else hashlib.sha256(path.read_bytes()).hexdigest()\n if digest != expected: raise RuntimeError(name+\": SHA256 mismatch\")\n print(name+\": verified\")')"
```

Nếu kiểm tra hash thất bại, không sử dụng file đó; kiểm tra lại nguồn tải và tải vào đường dẫn sạch. Không chạy script hoặc mã Python lấy từ repository trọng số.

## 4. Chạy pipeline

### Train: dự đoán và chấm đầy đủ

```bash
neorag run train --pipeline neorag.pipelines.evidence:EvidenceFeedbackPipeline
```

Lệnh trên gọi gateway khi cache chưa có. Trước khi chạy, kiểm tra ngân sách key và các đường dẫn đầu ra. Mặc định, chạy lại cùng split sẽ ghi đè predictions/ZIP cùng tên.

### Tạo bài nộp Public

```bash
neorag predict public_test --pipeline neorag.pipelines.evidence:EvidenceFeedbackPipeline
neorag validate public_test --predictions outputs/public_test_pred.jsonl
```

Đầu ra mặc định:

```text
outputs/public_test_pred.jsonl
outputs/public_test_submission.zip
```

ZIP của bản từng đạt 87,43 trên máy phát triển được lưu riêng với tên `public_test_evidence_v1_submission.zip`; file đó không được phân phối trong repository. Không coi ZIP vừa sinh lại là bản có cùng điểm trước khi chấm.

### Chấm predictions có sẵn

```bash
neorag evaluate train --predictions outputs/train_pred.jsonl
neorag evaluate train --predictions outputs/train_pred.jsonl --no-judge
```

`evaluate --no-judge` chạy retrieval và lexical, không gọi judge và không có generation score hay tổng điểm. Ngược lại, `run --no-judge` vẫn có thể gọi mô hình ở bước dự đoán. Xem `neorag <command> --help` để chọn đường dẫn riêng khi cần bảo toàn các lượt chạy cũ.

## 5. Chatbot demo

Chatbot Streamlit sử dụng cùng hợp đồng pipeline. Chỉ chạy khi muốn demo; không cần mở chatbot để tạo bài nộp.

Windows PowerShell:

```powershell
$env:NEORAG_PIPELINE = 'neorag.pipelines.evidence:EvidenceFeedbackPipeline'
python -m streamlit run app/chat.py
```

macOS/Linux:

```bash
NEORAG_PIPELINE=neorag.pipelines.evidence:EvidenceFeedbackPipeline python -m streamlit run app/chat.py
```

Chatbot hiển thị lịch sử hội thoại nhưng pipeline chỉ nhận câu hỏi hiện tại, không có suy luận đa lượt từ lịch sử. Các yêu cầu thật có thể tiêu ngân sách gateway.

## 6. Các nhánh kỹ thuật trong mã nguồn

Đây là các class thử nghiệm, **không phải các Git branch độc lập**. Không có class mới hơn nào tự động trở thành bản được chọn.

| Module | Nội dung | Trạng thái |
|---|---|---|
| `pipelines/baseline.py` | Dense retrieval và prompt tiếng Việt | Baseline, nền tảng để so sánh |
| `pipelines/hybrid.py` | Dense + BM25, biến thể metadata | Thành phần retrieval |
| `pipelines/reranked.py` | FlashRank/BGE, phối hợp điểm và diversity | BGE được dùng trong bản chọn |
| `pipelines/feedback.py` | Truy vấn bổ sung, xếp hạng theo độ phủ | Được dùng trong bản chọn |
| `pipelines/evidence.py` | Đối chiếu trích dẫn, xếp hạng theo bằng chứng; các biến thể hậu kiểm/nén | `EvidenceFeedbackPipeline` được chọn; biến thể khác chưa được chọn |
| `pipelines/decomposed.py` | Tách truy vấn theo khía cạnh | Thử nghiệm |
| `pipelines/scoped.py` | Scope/cards, truy tìm theo yêu cầu, chọn khẳng định trước khi viết | Thử nghiệm; chưa vượt điều kiện chọn |
| `pipelines/learned.py` | Cross-encoder học từ nhãn train | Thử nghiệm; chưa có mô hình cuối được chọn |
| `scripts/train_reranker.py` | Huấn luyện pairwise với hard negatives, đối chiếu theo nhóm nguồn | Công cụ nghiên cứu, không cần để chạy Evidence-v1 |

### Thử nghiệm học reranker

Chỉ dùng câu hỏi và nhãn nguồn của train; không dùng đáp án Public/Private Test. Model nền: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, snapshot `1427fd652930e4ba29e8149678df786c240d8825`. Cập nhật hai encoder block cuối và classification head, dùng pairwise loss giữa nguồn đúng và hard negative.

- 122 câu có nhãn nguồn, 196 cặp câu hỏi–nguồn đúng.
- 3 folds; các câu có chung nguồn đúng được gom cùng nhóm, kể cả quan hệ bắc cầu.
- Cùng corpus được sử dụng khi truy xuất; train đã được phân tích trước đó, nên không tuyên bố đây là kiểm chứng hoàn toàn độc lập.
- Retrieval score ngoài fold của model nhỏ tăng từ 78,29 lên 79,00, chưa đạt ngưỡng và chưa cạnh tranh với hệ thống Evidence-v1; **không huấn luyện model cuối để thay bản chọn**.

Để tự chạy lại, cần GPU CUDA và đủ dung lượng. Trên Windows, cài PyTorch CUDA tương thích trước; `python scripts/install.py --local` có cơ chế chọn wheel theo driver. Sau đó:

```bash
python -m pip install -e '.[train]'
python -I scripts/train_reranker.py
```

Script không gọi gateway BTC; tải trọng số công khai và lưu checkpoint tại `.cache/trained-reranker-v1/`, báo cáo tại `outputs/trained_reranker_v1/`. Mỗi cấu hình mới nên dùng `--work-dir` và `--report-dir` riêng. Xem `--help` trước khi chạy.

## 7. Kiểm thử và đánh giá

```bash
python -m pytest
python -m ruff check .
```

Tests của pipeline dùng FakeLLM/FakeEmbedder và monkeypatch, không cần key thật hoặc tải trọng số để kiểm tra luồng RAG. Một số test tích hợp thư viện tùy chọn được bỏ qua nếu chưa cài dependency tương ứng. Lượt kiểm tra phát triển gần nhất: **211 tests passed**, Ruff không báo lỗi.

Các pilot đã được sử dụng nhiều lần, nên kết quả thử nghiệm chỉ là bằng chứng phát triển. Chỉ cân nhắc thay pipeline sau khi kiểm tra hồi quy, chấm trên toàn train bằng bộ chấm nguyên bản và xác thực bài nộp. Repository không sửa `configs/eval.toml` hay mã nguồn bộ chấm để tăng điểm.

## 8. Cấu trúc repository

```text
src/neorag/
  pipelines/       Baseline, retrieval, evidence và các biến thể nghiên cứu
  metrics/         Metric retrieval, lexical và LLM judge của starter kit
  cli.py           CLI dự đoán, validate, đóng gói và đánh giá
  data.py          Hợp đồng corpus, câu hỏi và prediction
  llm.py           OpenAI-compatible gateway và cache SQLite
scripts/           Cài đặt và huấn luyện reranker
app/chat.py        Chatbot Streamlit mẫu
configs/eval.toml  Cấu hình bộ chấm của starter kit
data/              Corpus và các split được phép theo dõi trong Git
tests/             Kiểm thử offline
outputs/           Kết quả cục bộ, bị Git bỏ qua
.cache/            Cache gateway và model cục bộ, bị Git bỏ qua
```

Không commit `.env`, API key, outputs, trọng số, cache hoặc đề Private Test. Mọi `retrieved_ids` phải là ID số nguyên gốc trong `data/corpus.jsonl`, không phải ID chunk tự tạo.

## 9. Tài liệu và nguồn gốc

Dự án kế thừa starter kit tại [hoanghero125/NeoRAG-Cup-2026](https://github.com/hoanghero125/NeoRAG-Cup-2026). Các tài liệu dưới đây thuộc phần hướng dẫn của BTC, không phải tuyên bố dự án này là bộ chấm hay repository chính thức:

- [Hướng dẫn sử dụng](docs/guide.md)
- [Quy định cuộc thi](docs/rules.md)
- [Gateway](docs/gateway.md)
- [Định dạng và quy trình nộp bài](docs/submission.md)
- [Định nghĩa metric và cách chấm](docs/evaluation.md)
- [Dữ liệu và nguồn tham khảo](data/README.md)

Mã nguồn giữ [giấy phép MIT](LICENSE). Dữ liệu chỉ được sử dụng cho mục đích học tập trong khuôn khổ NeoRAG Cup 2026; bản quyền nội dung gốc thuộc các CLB và các nguồn được ghi trong `data/sources.csv`. Khi chia sẻ repository, bảo đảm người nhận được cấp quyền phù hợp và tuân thủ quy định dữ liệu của BTC.
