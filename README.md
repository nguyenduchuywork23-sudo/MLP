# Xây dựng MLP phân lớp Student Performance

Repo này cài đặt mạng nơ-ron nhân tạo đa lớp MLP từ đầu bằng `numpy` để phân lớp bộ dữ liệu `Student performance.csv`. Bài làm không dùng `sklearn`, `tensorflow`, `keras`, `pytorch` hoặc mô hình/classifier ML có sẵn.

**Sinh viên:** Nguyễn Đức Huy  
**Mã sinh viên:** BIT220079

## Cấu trúc file

- `mlp_student_performance.py`: code nguồn chính, gồm đọc dữ liệu, tiền xử lý, cài đặt MLP, huấn luyện, đánh giá, vẽ biểu đồ và tạo báo cáo.
- `Student performance.csv`: dữ liệu Student Performance.
- `results_mlp_student_performance.csv`: bảng kết quả thực nghiệm.
- `loss_best_config.png`: biểu đồ train/test loss của cấu hình tốt nhất.
- `loss_all_configs.png`: biểu đồ train loss của toàn bộ cấu hình.
- `test_accuracy_comparison.png`: biểu đồ cột so sánh Test accuracy của các cấu hình.
- `test_loss_comparison.png`: biểu đồ cột so sánh Test loss của các cấu hình.
- `confusion_matrix_best.png`: heatmap ma trận nhầm lẫn của cấu hình tốt nhất.
- `bao_cao_mlp_student_performance.md`: báo cáo dạng Markdown.
- `bao_cao_mlp_student_performance.pdf`: báo cáo PDF.

## Cài thư viện

```bash
pip install numpy pandas matplotlib
```

## Cách chạy

```bash
python mlp_student_performance.py
```

Sau khi chạy, chương trình sẽ tạo lại các file đầu ra:

- `results_mlp_student_performance.csv`
- `loss_best_config.png`
- `loss_all_configs.png`
- `test_accuracy_comparison.png`
- `test_loss_comparison.png`
- `confusion_matrix_best.png`
- `bao_cao_mlp_student_performance.md`
- `bao_cao_mlp_student_performance.pdf`

## MLP tự cài đặt

Mô hình trong `mlp_student_performance.py` tự cài đặt các thành phần chính:

- Forward propagation.
- Hàm kích hoạt ReLU cho lớp ẩn.
- Softmax cho lớp đầu ra.
- Cross Entropy Loss.
- Backpropagation.
- Mini-batch Gradient Descent.
- L2 regularization nhẹ.
- Learning rate decay đơn giản.

Các bước tiền xử lý như chia train/validation/test, Min-Max Scaling, one-hot encoding categorical, accuracy và confusion matrix cũng được tự cài đặt, không dùng hàm có sẵn từ thư viện học máy.

## Thực nghiệm

Chương trình thử nhiều cấu hình MLP với hai kiểu tiền xử lý:

- `Min-Max`: dùng trực tiếp các giá trị số sau khi chuẩn hóa bằng min/max fit trên train.
- `One-hot`: mã hóa các đặc trưng categorical bằng category fit trên train, sau đó transform validation/test theo cùng category.

Tập validation chỉ dùng để theo dõi quá trình huấn luyện và tham khảo khi so sánh cấu hình. Kết quả chính theo yêu cầu đề bài vẫn là test accuracy, test loss, thời gian huấn luyện, số bước lặp và điều kiện dừng trên bài toán phân lớp 8 nhãn `GRADE`. Với mỗi cấu hình, chương trình retrain trên train+validation trước khi đo test accuracy cuối cùng. Toàn bộ kết quả trong CSV, biểu đồ và báo cáo được sinh ra từ lệnh chạy thật, không sửa tay số liệu.
