# Ghi Chú Đánh Giá Bộ Câu Hỏi Kiểm Tra

**Người đánh giá:** Huỳnh Khải Huy
**Ngày:** 2026-05-12
**Bộ câu hỏi:** `phase-a/testset_v1.csv` (50 câu hỏi)

Đã đọc và đánh giá 12 câu hỏi đầu tiên. Các chỉnh sửa được đánh dấu **[EDITED]**.

---

## Câu hỏi 1 (single_hop_specific_query_synthesizer)

**Q:** Công ty ABC Việt Nam được thành lập khi nào?

**GT:** Công ty Cổ phần Công nghệ ABC Việt Nam được thành lập theo Giấy chứng nhận đăng ký doanh nghiệp số 0312345678 do Sở Kế hoạch và Đầu tư Thành phố Hồ Chí Minh cấp ngày 15 tháng 03 năm 2015.

**Đánh giá:** OK — Câu hỏi rõ ràng, ground truth chính xác, dễ kiểm chứng từ tài liệu.

---

## Câu hỏi 2 (single_hop_specific_query_synthesizer)

**Q (gốc):** Can you provide details about the location of the headquarters of Công ty Cổ phần Công nghệ ABC Việt Nam, specifically mentioning the Tòa nhà Bitexco Financial Tower?

**Q [EDITED]:** Trụ sở chính của Công ty Cổ phần Công nghệ ABC Việt Nam đặt tại đâu?

**GT:** Trụ sở chính của Công ty đặt tại Tầng 10, Tòa nhà Bitexco Financial Tower, 2 Hải Triều, Quận 1, TP. Hồ Chí Minh.

**Đánh giá:** **[EDITED]** — Câu hỏi gốc bằng tiếng Anh và quá dài dòng, đã chuyển sang tiếng Việt tự nhiên hơn. Ground truth điều chỉnh cho ngắn gọn hơn.

---

## Câu hỏi 3 (single_hop_specific_query_synthesizer)

**Q (gốc):** What accounting standards are used in financial reporting in Việt Nam?

**Q [EDITED]:** Báo cáo tài chính của công ty được lập theo chuẩn mực kế toán nào?

**GT:** Báo cáo tài chính được lập theo Chuẩn mực Kế toán Việt Nam (VAS) và các quy định của pháp luật Việt Nam có liên quan.

**Đánh giá:** **[EDITED]** — Câu gốc bằng tiếng Anh và hỏi chung chung; đã chỉnh sang tiếng Việt và gắn với ngữ cảnh tài liệu cụ thể.

---

## Câu hỏi 4 (single_hop_specific_query_synthesizer)

**Q:** What are the key financial indicators for Tập đoàn Công nghệ XYZ International in 2023, and how do they reflect the company's performance?

**GT:** Các chỉ số tài chính quan trọng năm 2023 gồm: lợi nhuận gộp 30,5%, lợi nhuận ròng 2,7%, ROA 3,3%, ROE 5,2%, D/E ratio 0,59 lần, current ratio 1,91 lần, quick ratio 1,72 lần, vòng quay phải thu 8,1 lần/năm.

**Đánh giá:** OK — Câu hỏi tổng hợp nhiều chỉ số, phù hợp để kiểm tra multi-fact retrieval. Tên tập đoàn trong câu hỏi là "XYZ International" nhưng tài liệu ghi "Tập đoàn Công nghệ XYZ International (Singapore)" — vẫn khớp.

---

## Câu hỏi 5 (single_hop_specific_query_synthesizer)

**Q:** Báo cáo tài chính của công ty phản ánh tình hình tài chính tại ngày 31/12/2023 như thế nào?

**GT:** Báo cáo tài chính phản ánh trung thực và hợp lý tình hình tài chính của Công ty tại ngày 31/12/2023 và kết quả hoạt động kinh doanh trong năm tài chính kết thúc cùng ngày, phù hợp với Chuẩn mực Kế toán Việt Nam.

**Đánh giá:** OK — Câu hỏi tự nhiên, ground truth trực tiếp từ ý kiến kiểm toán viên. Phù hợp để đo faithfulness.

---

## Câu hỏi 6 (single_hop_specific_query_synthesizer)

**Q:** Ai là kiểm toán viên trong báo cáo tài chính của Công ty?

**GT:** Kiểm toán viên là Nguyễn Văn An, Chứng chỉ Kiểm toán viên số 2345/KTV.

**Đánh giá:** OK — Câu hỏi rõ, ground truth ngắn gọn, dễ đánh giá đúng/sai. Đây là loại câu hỏi factual kiểm tra độ chính xác của retrieval.

---

## Câu hỏi 7 (single_hop_specific_query_synthesizer)

**Q:** Nghị định này có áp dụng cho cơ quan, tổ chức, cá nhân nào ở Việt Nam không?

**GT:** Nghị định áp dụng đối với cơ quan, tổ chức, cá nhân Việt Nam và cơ quan, tổ chức, cá nhân nước ngoài trực tiếp tham gia hoặc có liên quan đến hoạt động xử lý dữ liệu cá nhân tại Việt Nam.

**Đánh giá:** OK — Ground truth đầy đủ, bao gồm cả trường hợp tổ chức nước ngoài — điểm quan trọng trong Nghị định 13/2023.

---

## Câu hỏi 8 (single_hop_specific_query_synthesizer)

**Q (gốc):** Cơ quan, tổ chức, cá nhân nào ở Việt Nam phải tuân thủ quy định về bảo vệ dữ liệu cá nhân?

**Đánh giá:** **[EDITED — DELETED]** — Câu hỏi này gần như trùng lặp với câu hỏi 7 (cùng hỏi về đối tượng áp dụng của Nghị định 13). Ground truth cũng giống nhau. Đề xuất loại bỏ để tránh câu hỏi trùng trong bộ kiểm tra.

---

## Câu hỏi 9 (single_hop_specific_query_synthesizer)

**Q:** Sự đồng ý là gì trong bối cảnh xử lý dữ liệu cá nhân?

**GT:** Sự đồng ý của chủ thể dữ liệu là sự thể hiện rõ ràng, tự nguyện, cụ thể, có thông tin và không mơ hồ của chủ thể dữ liệu về việc cho phép xử lý dữ liệu cá nhân của họ.

**Đánh giá:** OK — Định nghĩa pháp lý chính xác, quan trọng để đo khả năng RAG trích xuất định nghĩa thuật ngữ pháp lý.

---

## Câu hỏi 10 (single_hop_specific_query_synthesizer)

**Q:** Bên Xử lý dữ liệu cá nhân là gì và vai trò của nó trong việc xử lý dữ liệu?

**GT:** Bên Xử lý dữ liệu cá nhân là tổ chức, cá nhân thực hiện việc xử lý dữ liệu thay mặt cho Bên Kiểm soát dữ liệu, thông qua một hợp đồng hoặc thỏa thuận với Bên Kiểm soát dữ liệu cá nhân.

**Đánh giá:** OK — Câu hỏi định nghĩa rõ ràng. Ground truth chính xác, không có thông tin thừa.

---

## Câu hỏi 11 (single_hop_specific_query_synthesizer)

**Q:** Bên Kiểm soát dữ liệu cá nhân có nghĩa vụ gì đối với chủ thể dữ liệu?

**GT (gốc):** Bên Kiểm soát dữ liệu cá nhân có nghĩa vụ bảo vệ dữ liệu cá nhân của chủ thể dữ liệu và tôn trọng quyền của họ, bao gồm việc cung cấp đầy đủ và chính xác thông tin liên quan đến hoạt động xử lý dữ liệu cá nhân.

**GT [EDITED]:** Bên Kiểm soát dữ liệu cá nhân có nghĩa vụ bảo vệ dữ liệu cá nhân của chủ thể dữ liệu, tôn trọng quyền của họ, thực hiện các biện pháp bảo mật kỹ thuật và tổ chức phù hợp, đồng thời thông báo cho cơ quan có thẩm quyền khi xảy ra vi phạm dữ liệu trong vòng 72 giờ.

**Đánh giá:** **[EDITED]** — Ground truth gốc quá ngắn, bỏ sót nghĩa vụ thông báo vi phạm 72 giờ (được quy định rõ trong Nghị định 13). Đã bổ sung để ground truth đầy đủ hơn.

---

## Câu hỏi 12 (single_hop_specific_query_synthesizer)

**Q:** Nghị định quy định những quyền gì của chủ thể dữ liệu?

**GT:** Nghị định quy định 11 quyền: quyền được biết, quyền đồng ý, quyền truy cập, quyền rút lại sự đồng ý, quyền xóa dữ liệu, quyền hạn chế xử lý, quyền cung cấp dữ liệu, quyền phản đối xử lý, quyền khiếu nại/tố cáo/khởi kiện, quyền yêu cầu bồi thường thiệt hại, và quyền tự bảo vệ.

**Đánh giá:** OK — Câu hỏi quan trọng nhất trong bộ kiểm tra về Nghị định 13/2023. Ground truth liệt kê đủ 11 quyền, phù hợp để kiểm tra context recall.

---

## Tóm Tắt Đánh Giá

| Câu # | Trạng thái | Lý do |
| --- | --- | --- |
| 1 | OK | Factual, chính xác |
| 2 | EDITED | Chuyển từ tiếng Anh sang tiếng Việt, rút gọn |
| 3 | EDITED | Chuyển từ tiếng Anh sang tiếng Việt, gắn ngữ cảnh |
| 4 | OK | Multi-fact, phù hợp |
| 5 | OK | Factual từ ý kiến kiểm toán |
| 6 | OK | Factual ngắn gọn |
| 7 | OK | Định nghĩa đối tượng áp dụng đầy đủ |
| 8 | DELETED | Trùng lặp với câu 7 |
| 9 | OK | Định nghĩa thuật ngữ pháp lý chính xác |
| 10 | OK | Định nghĩa rõ ràng |
| 11 | EDITED | Bổ sung nghĩa vụ thông báo vi phạm 72 giờ vào GT |
| 12 | OK | Liệt kê đủ 11 quyền |

**Tổng:** 9 OK, 3 EDITED (trong đó 1 DELETED), không có câu hỏi nào cần loại bỏ hoàn toàn ngoài câu 8.
