# Báo cáo Cá nhân — Thành viên #3 (AI Engineer / Multi-Judge Specialist)

## 1. Thông tin cá nhân
- Họ và tên: Nguyễn Thị Quỳnh Trang
- MSSV: 2A202600406
- Vai trò trong nhóm: Thiết kế và triển khai Multi-Judge Consensus Engine, tối ưu chi phí đánh giá, kiểm thử độ ổn định của pipeline chấm điểm.

## 2. Mục tiêu cá nhân trong Lab
Trong Lab Day 14, mục tiêu kỹ thuật cá nhân của tôi là xây dựng một hệ thống chấm điểm đa giám khảo đủ mạnh cho production theo 4 yêu cầu:
1. Chấm song song bằng nhiều mô hình để giảm thiên lệch đơn lẻ.
2. Có cơ chế xử lý xung đột tự động khi các Judge bất đồng mạnh.
3. Có kiểm tra Position Bias để tăng độ tin cậy của kết quả.
4. Có chiến lược tối ưu chi phí mà vẫn giữ chất lượng đánh giá.

## 3. Engineering Contribution (đóng góp kỹ thuật chính)

### 3.1 Phạm vi module phụ trách
- Module chính: engine/llm_judge.py
- Thành phần đã hoàn thiện:
1. Gọi Judge song song GPT-4o-mini và Claude Haiku bằng async.
2. Rubrics 5 tiêu chí: Accuracy, Faithfulness, Completeness, Professionalism, Safety.
3. Conflict Resolution:
- Nếu chênh lệch điểm <= 1: lấy trung bình.
- Nếu chênh lệch điểm > 1: gọi GPT-4o full làm tie-breaker.
4. Position Bias Check bằng cách tráo vị trí A/B và chấm lại.
5. Claude JSON Extraction (vì Claude không ép response_format JSON như OpenAI).
6. Fast Mode để giảm chi phí: Claude chấm trước, short-circuit ở ngưỡng cực trị.

### 3.2 Kết quả triển khai chi tiết
1. Xây dựng output có cấu trúc ổn định cho runner:
- final_score
- agreement_rate
- individual_scores
- conflict_resolved_by
- fast_mode, fast_mode_short_circuit

2. Hardening parser để tránh crash:
- Hỗ trợ parse JSON trực tiếp.
- Hỗ trợ parse JSON trong markdown fence.
- Hỗ trợ tìm object JSON đầu tiên trong chuỗi tự do.

3. Tương thích ngược khi tích hợp:
- Bổ sung alias conflict_resolved để tránh lỗi KeyError do khác key naming.
- Bổ sung cost_usd với giá trị mặc định None để không phá vỡ luồng cũ.

4. Tối ưu luồng chạy thử:
- Bổ sung in trạng thái fast_mode và fast_mode_short_circuit trong khối test cuối file để quan sát rõ có tiết kiệm call GPT hay không.

### 3.3 Giá trị mang lại cho hệ thống
1. Tăng độ khách quan:
Không còn phụ thuộc một Judge duy nhất.

2. Tăng độ ổn định kỹ thuật:
Parser có khả năng chịu lỗi format đầu ra của LLM tốt hơn.

3. Tăng hiệu quả chi phí:
Fast Mode giúp giảm số lần gọi GPT-4o-mini ở các case quá rõ ràng.

4. Tăng tính sẵn sàng production:
Có conflict handling, bias checking, metadata output rõ ràng để audit.

## 4. Technical Depth (độ sâu kỹ thuật)

### 4.1 Logic đồng thuận đa giám khảo
Với điểm từ GPT là S_gpt và điểm từ Claude là S_claude:
- Score gap: |S_gpt - S_claude|
- Nếu gap <= 1: final_score = (S_gpt + S_claude) / 2
- Nếu gap > 1: final_score = điểm của tie-breaker GPT-4o full

Điểm đồng thuận (agreement_rate) được tính theo khoảng cách điểm từng tiêu chí và overall, sau đó chuẩn hóa về [0, 1].

### 4.2 Ý nghĩa của Position Bias
Position Bias xảy ra khi cùng một nội dung nhưng do đặt ở vị trí A hoặc B mà Judge chấm khác nhau đáng kể.

Cách kiểm tra đã triển khai:
1. Chấm cặp (A, B).
2. Chấm cặp đã swap (B, A).
3. Quy chiếu winner về cùng hệ trục và so sánh.

Nếu kết quả không nhất quán sau quy chiếu thì đánh dấu có bias.

### 4.3 Trade-off Chi phí và Chất lượng
Fast Mode được thiết kế theo tư duy tiered judging:
1. Chạy Claude Haiku trước (chi phí thấp hơn).
2. Nếu điểm cực trị (<=2 hoặc >=4.5) thì chốt luôn.
3. Chỉ gọi GPT-4o-mini khi cần ý kiến thứ hai.

Ý nghĩa:
- Với case rõ ràng, tiết kiệm API calls.
- Với case biên, vẫn đảm bảo chất lượng nhờ judge bổ sung và tie-breaker.

### 4.4 MRR, Cohen's Kappa và liên hệ với module cá nhân
Dù tôi phụ trách chính Multi-Judge, tôi nắm rõ các chỉ số nền tảng để phối hợp liên module:

1. MRR (Mean Reciprocal Rank):
MRR = (1/N) * tổng (1/rank_i)
- Đo chất lượng retrieval theo vị trí tài liệu đúng đầu tiên.
- Retrieval tốt giúp giảm hallucination, gián tiếp nâng điểm judge.

2. Cohen's Kappa:
Kappa = (P_o - P_e) / (1 - P_e)
- P_o: tỷ lệ đồng ý quan sát được.
- P_e: tỷ lệ đồng ý ngẫu nhiên kỳ vọng.
- Kappa phản ánh độ đồng thuận mạnh hơn agreement rate đơn giản, là hướng nâng cấp kế tiếp.

## 5. Problem Solving (xử lý sự cố trong quá trình làm)

### Vấn đề 1: Claude không hỗ trợ response_format JSON tự động
- Triệu chứng: Kết quả Claude có thể lẫn văn bản ngoài JSON.
- Nguyên nhân: API behavior khác OpenAI.
- Cách xử lý:
1. Viết hàm trích text từ content blocks.
2. Viết parser nhiều tầng: direct JSON, fenced JSON, first-object JSON.
- Kết quả: Giảm lỗi parse, tăng độ ổn định pipeline.

### Vấn đề 2: KeyError khi chạy test do lệch key output
- Triệu chứng: Truy cập result["conflict_resolved"] gây lỗi.
- Nguyên nhân: Output thực tế dùng key conflict_resolved_by.
- Cách xử lý:
1. Chuẩn hóa truy cập bằng get có fallback.
2. Bổ sung alias conflict_resolved để tương thích ngược.
- Kết quả: Luồng test chạy ổn định, không văng runtime error.

### Vấn đề 3: Bật fast_mode nhưng kết quả nhìn như không đổi
- Triệu chứng: Score gần như giống chế độ thường.
- Nguyên nhân:
1. Có lần gọi chưa truyền fast_mode=True.
2. Dù bật fast_mode, case không rơi vào ngưỡng short-circuit nên vẫn gọi GPT.
- Cách xử lý:
1. In rõ fast_mode và fast_mode_short_circuit.
2. Giải thích cơ chế theo ngưỡng và output để tránh hiểu nhầm.
- Kết quả: Dễ quan sát chính xác khi nào tiết kiệm được chi phí.

## 6. Minh chứng làm việc (evidence)
1. Triển khai đầy đủ engine/llm_judge.py theo đúng tiêu chí bài lab.
2. Có xử lý conflict resolution và tie-breaker bằng GPT-4o full.
3. Có Position Bias Check bằng swap A/B.
4. Có Fast Mode để tối ưu cost.
5. Có hardening parser và compatibility layer để chống lỗi runtime.

## 7. Tự đánh giá theo rubric cá nhân

### 7.1 Engineering Contribution (15/15 mục tiêu)
- Đóng góp trực tiếp vào module phức tạp nhất của hệ thống đánh giá: Multi-Judge Engine.
- Có triển khai async, xung đột điểm, tie-breaker, bias check, fast mode.
- Có thể chứng minh bằng code và lịch sử commit.

### 7.2 Technical Depth (15/15 mục tiêu)
- Nắm và áp dụng đúng trade-off Cost vs Quality.
- Hiểu sâu Position Bias và cơ chế kiểm tra thực nghiệm.
- Trình bày rõ nguyên lý MRR, Cohen's Kappa và ý nghĩa trong hệ đánh giá.

### 7.3 Problem Solving (10/10 mục tiêu)
- Chủ động xử lý các lỗi thực tế trong tích hợp API và schema output.
- Giải quyết triệt để lỗi runtime và lỗi nhận thức khi đọc kết quả fast mode.
- Đảm bảo pipeline ổn định và dễ debug hơn.

Kỳ vọng điểm cá nhân: 38-40/40 nếu kèm commit evidence rõ ràng.

## 8. Bài học rút ra
1. Hệ thống đánh giá AI đáng tin phải có nhiều lớp bảo vệ: multi-judge, conflict handling, bias checking.
2. Độ ổn định parser và chuẩn hóa output quan trọng không kém thuật toán chấm điểm.
3. Tối ưu chi phí hiệu quả nhất là phân tầng quyết định, không phải giảm model một cách cứng nhắc.
4. Báo cáo kỹ thuật muốn thuyết phục cần gắn chặt giữa kiến trúc, số liệu và sự cố đã xử lý.

## 9. Kế hoạch nâng cấp sau Lab
1. Thêm logging token usage/cost thật theo từng model call thay vì để None.
2. Tính Cohen's Kappa trên toàn bộ benchmark set để có thước đo đồng thuận chặt chẽ hơn.
3. Bổ sung dashboard phân tích bias theo từng loại câu hỏi (easy/medium/hard/adversarial).
4. Kết nối đầy đủ với summary report để thể hiện rõ hiệu quả tiết kiệm của fast mode.

---
Tôi cam kết phần đóng góp trên là phần tôi trực tiếp thiết kế, triển khai và kiểm thử trong bài Lab này.
