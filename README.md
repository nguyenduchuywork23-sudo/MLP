# Xây dựng MLP phân lớp Student Performance

Repo này cài đặt mạng nơ-ron nhân tạo đa lớp MLP từ đầu bằng `numpy` để phân lớp bộ dữ liệu `Student performance.csv`. Bài làm không dùng `sklearn`, `tensorflow`, `keras`, `pytorch` hoặc mô hình/classifier ML có sẵn.

## Cấu trúc file

- `mlp_student_performance.py`: code nguồn chính, gồm đọc dữ liệu, tiền xử lý, cài đặt MLP, huấn luyện, đánh giá, vẽ biểu đồ và tạo báo cáo.
- `Student performance.csv`: dữ liệu Student Performance.
- `results_mlp_student_performance.csv`: bảng kết quả thực nghiệm.
- `loss_best_config.png`: biểu đồ train/test loss của cấu hình tốt nhất.
- `loss_all_configs.png`: biểu đồ train loss của toàn bộ cấu hình.
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

Các bước tiền xử lý như chia train/test, Min-Max Scaling, one-hot encoding, accuracy và confusion matrix cũng được tự cài đặt, không dùng hàm có sẵn từ thư viện học máy.
