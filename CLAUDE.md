# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Bối cảnh và ràng buộc

Repository là starter kit Python và bộ chấm chính thức của NeoRAG Cup 2026: hệ thống RAG tiếng Việt về cộng đồng S2B (ProPTIT, MPC, GCC), kèm pipeline baseline, dữ liệu, công cụ tạo bài nộp và chatbot Streamlit mẫu. Tài liệu và thông báo CLI chủ yếu dùng tiếng Việt.

Các tài liệu cần đối chiếu khi thay đổi hành vi:

- `docs/guide.md`: cài đặt, chạy pipeline và mở rộng hệ thống.
- `docs/rules.md`: giới hạn công nghệ và quy định cuộc thi.
- `docs/evaluation.md`: ý nghĩa metric và cách tổng hợp điểm.
- `docs/submission.md`, `data/README.md`: hợp đồng bài nộp và dữ liệu.

Mọi bước dùng mô hình sinh văn bản phải gọi `claude-haiku-5-5` qua gateway tương thích OpenAI của BTC. Repository dùng OpenAI SDK, không dùng Anthropic SDK. Không truyền `temperature` vì gateway không hỗ trợ. Embedding mặc định là `bge-m3`; được phép dùng embedding/reranker mã nguồn mở không sinh văn bản chạy cục bộ.

`configs/eval.toml` cùng mã nguồn bộ chấm xác định cách chấm chính thức; sửa bản cục bộ không thay đổi bộ chấm trên hệ thống thi. Không báo cáo điểm từ bộ chấm đã sửa như điểm chính thức. Không hard-code đáp án hoặc ID truy xuất cho câu hỏi test, không gán nhãn thủ công Public/Private Test; quy định cũng cấm sửa mã nguồn trong ngày Private Test.

## Cài đặt và các lệnh

Yêu cầu Python 3.10+. Chạy từ thư mục gốc repository trong môi trường đã kích hoạt. Trên Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python scripts/install.py
```

Trên macOS/Linux, tạo bằng `python3 -m venv .venv`, kích hoạt bằng `source .venv/bin/activate`. Installer cài package editable với extras `app,dev`; cách thay thế là `python -m pip install -r requirements.txt`.

Nếu cần embedding/reranker cục bộ, dùng `python scripts/install.py --local`. Script chọn bản PyTorch CUDA tương thích driver NVIDIA, tránh trường hợp pip cài bản CPU trên Windows.

```bash
# Toàn bộ test; cấu hình pytest mặc định dùng tests/ và -q
python -m pytest

# Một file, một test hoặc lọc theo tên
python -m pytest tests/test_pipeline.py
python -m pytest tests/test_cli.py::test_run_does_predict_and_evaluate
python -m pytest -k retrieval

# Lint theo cấu hình trong pyproject.toml
python -m ruff check .

# Đóng gói wheel khi cần; build backend là Hatchling
python -m pip wheel --no-deps . --wheel-dir dist

# CLI và chatbot mẫu
neorag --help
python -m streamlit run app/chat.py
```

Tests dùng `FakeLLM`, `FakeEmbedder` trong `tests/conftest.py` và monkeypatch các lời gọi gateway/nạp model; không cần key thật hoặc tải model. Không có bước build frontend riêng.

Sao chép `.env.example` thành `.env` rồi điền `OPENAI_API_KEY` trước khi gọi gateway thật. Baseline đọc `OPENAI_BASE_URL`, `LLM_MODEL`, `EMBEDDING_MODEL`; judge có thể dùng `JUDGE_API_KEY`, `JUDGE_BASE_URL` riêng. Các lời gọi thật tiêu thụ ngân sách giới hạn của cuộc thi.

```bash
# Dự đoán, kiểm tra, đóng gói và chấm nếu có ground truth
neorag run train
neorag run train --pipeline my_solution.pipeline:MyPipeline

# Tạo bài nộp và kiểm tra file
neorag predict public_test
neorag validate public_test --predictions outputs/public_test_pred.jsonl

# Chấm file predictions có sẵn, không gọi gateway
neorag evaluate train --no-judge

# Chấm đầy đủ bằng judge chính thức, cần key gateway
neorag evaluate train
```

`neorag run train --no-judge` chỉ bỏ bước judge: pipeline dự đoán vẫn gọi model. Chế độ `--no-judge` báo cáo retrieval và lexical nhưng không có generation score hoặc điểm tổng. Dùng `--help` của từng lệnh để xem các tùy chọn đường dẫn như `--out`, `--predictions`, `--gold`.

## Kiến trúc và luồng xử lý

Package nằm trong `src/neorag`, entry point console là `neorag.cli:main`. CLI điều phối ba lớp tách biệt: dự đoán, xác thực bài nộp và đánh giá. Bộ chấm đọc file kết quả, không thực thi pipeline của thí sinh.

### Hợp đồng pipeline

`pipelines/base.py` định nghĩa `RAGPipeline`, `RAGOutput` và cơ chế nạp động bằng chuỗi `module.path:ClassName`. Class tùy chỉnh phải kế thừa `RAGPipeline` và khởi tạo được không cần tham số khi dùng CLI/chatbot. Vòng đời: khởi tạo → `prepare()` một lần → `index(corpus)` một lần → `run(query)` cho từng câu hỏi. `prepare()` là tùy chọn; `run()` trả về câu trả lời và danh sách ID corpus theo thứ tự liên quan.

`pipelines/baseline.py` embed nguyên đoạn văn qua gateway, chuẩn hóa vector và xếp hạng cosine bằng NumPy. Baseline trả tối đa 10 ID, đưa 5 đoạn đầu vào một lần gọi sinh câu trả lời. Có thể inject `llm` và `embedder` để test offline. `embeddings.py` cung cấp local embedder, chỉ nạp sentence-transformers khi cần.

`predict.py` đo latency từng câu, chuyển đầu ra thành `Prediction`, loại ID trùng nhưng giữ thứ hạng và giới hạn số ID theo cấu hình. CLI ghi JSONL, kiểm tra hợp lệ rồi tạo ZIP bài nộp nếu không có lỗi. Đầu ra mặc định là `outputs/<split>_pred.jsonl` và `outputs/<split>_submission.zip`; chạy lại cùng split sẽ ghi đè các đường dẫn này.

`app/chat.py` dùng cùng hợp đồng pipeline, cache bước chuẩn bị/lập chỉ mục bằng `st.cache_resource`. Chọn class khác qua biến `NEORAG_PIPELINE`. Giao diện hiển thị lịch sử chat nhưng chỉ truyền câu hỏi hiện tại vào `pipeline.run`.

### Dữ liệu và bộ chấm

`data.py` quản lý các dataclass `Document`, `Query`, `Prediction`, đọc/ghi JSONL và xác thực bài nộp. ID đoạn văn là số nguyên; ID câu hỏi/prediction là chuỗi. Mỗi câu hỏi cần đúng một prediction, từ 1 đến 10 ID corpus hợp lệ không trùng, câu trả lời không rỗng và tối đa 2.000 ký tự. Reader chấp nhận UTF-8 BOM và bỏ qua trường dư.

Tên split trỏ tới `data/<split>/questions.jsonl`; CLI cũng hỗ trợ đường dẫn JSONL trực tiếp. Khi chấm theo thư mục split, ưu tiên `gold.jsonl` nếu có, nếu không dùng file questions. Ground truth được nhận diện qua `answer` không rỗng. Train có đáp án; Public/Private Test chỉ phát hành câu hỏi, nên `run` chỉ tạo bài nộp cho các tập này. Thư mục test trống có thể do chưa đến lịch phát hành.

Khi chunking hoặc bổ sung dữ liệu, `retrieved_ids` vẫn phải ánh xạ về ID gốc trong `data/corpus.jsonl`, loại trùng và giữ thứ hạng; không nộp ID chunk tự tạo.

`evaluate.py` nhận gold, predictions, corpus và cấu hình TOML, độc lập với pipeline. Giữ lớp chấm cùng các module data/config/text/metrics không phụ thuộc model nặng để chạy được trong môi trường chấm server. `metrics/retrieval.py` và `metrics/lexical.py` chạy cục bộ; `metrics/judge.py` gọi gateway song song theo số worker cấu hình. `text.py` tập trung chuẩn hóa Unicode/dấu thanh tiếng Việt và tách câu, nên thay đổi tại đây ảnh hưởng nhiều metric.

`configs/eval.toml` cấu hình báo cáo retrieval tại k=3,5,7; điểm tổng dùng Recall/MRR/NDCG tại k=5 và nhóm generation, trọng số hai nhóm bằng nhau. Câu hỏi không có đáp án được chấm abstention thay vì các metric generation thông thường. Đọc `docs/evaluation.md` trước khi thay đổi cách tổng hợp hoặc hành vi judge. Báo cáo gồm `report.md`, `report.json`, `per_query.jsonl` trong `outputs/eval_<prediction-stem>/`.

### Gateway, cache và đường dẫn

`llm.py` nạp `.env`, bọc chat/embedding theo OpenAI API và cache SQLite có đồng bộ luồng, khóa cache theo endpoint/model/request. Baseline dùng `.cache/pipeline.sqlite`, judge dùng `.cache/judge.sqlite`. Giữ cache khi thử nghiệm để không tiêu tốn lại ngân sách cho cùng lời gọi.

Đường dẫn corpus/config mặc định neo theo vị trí mã nguồn, còn outputs/cache tính theo thư mục làm việc hiện tại; chạy từ gốc repository để nhất quán. Outputs, cache, môi trường cục bộ, gold test và câu hỏi Private Test chưa phát hành được chủ động loại khỏi Git.
