# Báo cáo Cá nhân — Thành viên #2 (Data Engineer / Analyst)

## 1. Thông tin cá nhân
- **Họ và tên:** Đồng Văn Thịnh
- **MSSV:** 2A202600365
- **Vai trò trong nhóm:** Xây dựng bộ tập dữ liệu chuẩn (Golden Dataset), triển khai Synthetic Data Generation (SDG) và thực hiện Phân tích thất bại (Failure Analysis) sau Benchmark.

---

## 2. Mục tiêu cá nhân trong Lab

Mục tiêu kỹ thuật của tôi trong Lab Day 14 là cung cấp "nguyên liệu" chất lượng cao nhất cho toàn bộ hệ thống đánh giá:

1. **Chuẩn hóa dữ liệu nguồn:** Chuyển đổi các tài liệu thô thành bộ corpus chuẩn (`doc_001` - `doc_005`) để đảm bảo logic retrieval đồng nhất giữa các thành viên.
2. **Triển khai SDG Pipeline:** Xây dựng script tự động tạo 50+ test cases đa dạng độ khó bằng GPT-4o-mini, tối ưu chi phí token (Prompt tiếng Anh, Output tiếng Việt).
3. **Thử nghiệm bảo mật (Red Teaming):** Thiết kế các câu hỏi bẫy (Toxic Traps, Prompt Injection) để kiểm tra ngưỡng an toàn của hệ thống.
4. **Phân tích nguyên nhân gốc rễ:** Sử dụng kỹ thuật **5 Whys** để bóc tách các lỗi của Agent, từ đó đề xuất hướng cải thiện cho nhóm.

---

## 3. Engineering Contribution

### 3.1 Module phụ trách

| File | Nội dung đã implement / đóng góp |
| :--- | :--- |
| `data/synthetic_gen.py` | Toàn bộ logic sinh dữ liệu tự động, xử lý ID duy nhất và ánh xạ expected_retrieval_ids. |
| `data/golden_set.jsonl` | Tập dữ liệu 55 cases (30 Easy/Medium, 10 Hard, 15 Adversarial). |
| `analysis/failure_analysis.md` | Báo cáo phân tích 5 Whys, phân loại lỗi và đề xuất Action Plan dựa trên kết quả thực tế. |

### 3.2 Chi tiết triển khai `data/synthetic_gen.py`

**Chiến lược tối ưu Token:**
- Sử dụng `SYSTEM_PROMPT` bằng tiếng Anh để mô hình LLM hiểu sâu các khái niệm kỹ thuật (Adversarial, Ambiguous) nhưng ép output trả về tiếng Việt để khớp với corpus. Việc dùng prompt tiếng Anh giúp giảm khoảng 30% lượng token tiêu thụ so với prompt tiếng Việt dài dòng.

**Quản lý ID và Metadata:**
- Implement bộ đếm `doc_counts` toàn cục cho mỗi tài liệu để đảm bảo các ID như `doc_001_adversarial_001` không bao giờ bị trùng lặp khi chạy song song.
- Thiết lập quy tắc `expected_retrieval_ids = []` cho các câu hỏi Adversarial (vì đây là câu hỏi ngoài lề/bẫy), giúp metrics tính toán chính xác hơn.

---

## 4. Technical Depth

### 4.1 Phương pháp 5 Whys trong Failure Analysis
Tôi đã áp dụng phương pháp 5 Whys để phân tích 3 lỗi nghiêm trọng nhất sau khi chạy Benchmark:
- **Lỗi Hallucination:** Phát hiện ra LLM ưu tiên "Kiến thức nền" (Prior Knowledge) hơn là Context khi System Prompt thiếu các ràng buộc phủ định (Negative Constraints).
- **Lỗi Safety Guardrail:** Phát hiện Agent từ chối bẫy XSS quá máy móc khiến điểm Professionalism thấp. 
- **Lỗi Inaccuracy:** Chỉ ra sự xung đột giữa vai trò Line Manager và IT Admin trong tài liệu, đề xuất tăng `top_k` để bao phủ đủ context.

### 4.2 Metadata Mapping Contract
Để giải quyết bài toán Hit Rate bị 0% do Member 4 thực hiện Chunking, tôi đã đề xuất và thiết lập hệ thống **Metadata Mapping**. 
- Thay vì so sánh ID chunk ngẫu nhiên, hệ thống sẽ ánh xạ về `source_id: "doc_001"` được lưu trong metadata của mỗi chunk. Giải pháp này giúp Hit Rate phản ánh đúng khả năng tìm thấy tài liệu gốc.

---

## 5. Problem Solving

### Vấn đề 1: Trùng lặp ID khi sinh dữ liệu song song
- **Triệu chứng:** Khi chạy async nhiều độ khó cho cùng 1 doc, các task bắt đầu từ bộ đếm 0 tạo ra ID trùng.
- **Cách xử lý:** Thay đổi logic gán ID từ trong hàm worker ra ngoài hàm main, sử dụng dictionary để theo dõi bộ đếm ID cho từng tài liệu một cách tuần tự.

### Vấn đề 2: Lỗi Encoding trên môi trường Windows
- **Triệu chứng:** Script `check_lab.py` của instructor bị crash khi in các emoji (🔍, ✅) trên terminal Windows.
- **Cách xử lý:** Sử dụng `sys.stdout.reconfigure(encoding='utf-8')` để ép terminal xử lý Unicode, đảm bảo lead và các thành viên khác có thể kiểm tra bài nộp ổn định trên mọi hệ điều hành.

---

## 6. Minh chứng làm việc
- `data/golden_set.jsonl`: 55 test cases chất lượng, bao gồm 5 câu bẫy Red Teaming chuyên sâu.
- `analysis/failure_analysis.md`: Báo cáo dài 5414 bytes với phân tích kỹ thuật chi tiết.
- Git commits thể hiện việc refactor script SDG và sửa lỗi Unicode cho project.

---

## 7. Tự đánh giá theo rubric cá nhân

### Engineering Contribution (15/15)
Đã hoàn thành xuất sắc module SDG và trực tiếp xử lý các lỗi hệ thống (Encoding, ID Sync). Dữ liệu cung cấp là xương sống cho toàn bộ quá trình Benchmark.

### Technical Depth (13/15)
Hiểu sâu về các kỹ thuật phân tích lỗi và logic đánh giá retrieval. Tuy nhiên, chưa nghiên cứu sâu về các metric thống kê nâng cao như Cohen's Kappa để đóng góp thêm cho Member 3.

### Problem Solving (10/10)
Giải quyết triệt để 2 vấn đề ngăn cản tiến độ nhóm (ID duplication và Script crash on Windows).

---

## 8. Bài học rút ra
1. **Dữ liệu rác tạo ra kết quả rác (GIGO):** Bộ Golden Set nếu không được thiết kế kỹ (đặc biệt là phần Adversarial) sẽ làm mờ đi các điểm yếu thực sự của Agent.
2. **Kỹ năng phân tích quan trọng hơn số liệu:** Việc nhìn vào 10/50 câu sai và tự đặt câu hỏi "Tại sao" 5 lần mới thực sự giúp tìm ra hướng tối ưu cho hệ thống RAG.
3. **Giao tiếp (Communication) là chìa khóa:** Việc thống nhất ID tài liệu (`doc_001`) từ đầu đã tiết kiệm cho nhóm tôi hàng giờ đồng hồ sửa lỗi tích hợp.
