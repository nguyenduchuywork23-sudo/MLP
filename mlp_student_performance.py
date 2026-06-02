import csv
import sys
import textwrap
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


DATA_FILE = "Student performance.csv"
RESULTS_CSV = "results_mlp_student_performance.csv"
LOSS_BEST_PNG = "loss_best_config.png"
LOSS_ALL_PNG = "loss_all_configs.png"
REPORT_MD = "bao_cao_mlp_student_performance.md"
REPORT_PDF = "bao_cao_mlp_student_performance.pdf"

GRADE_NAMES = {
    0: "Fail",
    1: "DD",
    2: "DC",
    3: "CC",
    4: "CB",
    5: "BB",
    6: "BA",
    7: "AA",
}


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df


def split_features_labels(df):
    data = df.drop(columns=["STUDENT ID"]).copy()
    y = data["GRADE"].to_numpy(dtype=int)
    X = data.drop(columns=["GRADE"]).to_numpy(dtype=float)
    feature_names = list(data.drop(columns=["GRADE"]).columns)
    return X, y, feature_names


def stratified_train_test_split(X, y, test_size=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_indices = []
    test_indices = []

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        n_test = max(1, int(round(len(cls_indices) * test_size)))
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def fit_minmax_scaler(X_train):
    min_values = X_train.min(axis=0)
    max_values = X_train.max(axis=0)
    ranges = max_values - min_values
    ranges[ranges == 0] = 1.0
    return min_values, ranges


def transform_minmax(X, min_values, ranges):
    return (X - min_values) / ranges


def one_hot_encode(y, num_classes):
    encoded = np.zeros((len(y), num_classes))
    encoded[np.arange(len(y)), y] = 1.0
    return encoded


def accuracy_score_manual(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def confusion_matrix_manual(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)
    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1
    return matrix


class MLPClassifierScratch:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01, seed=42):
        self.input_size = input_size
        self.hidden_layers = list(hidden_layers)
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.weights = []
        self.biases = []
        self._initialize_parameters()

    def _initialize_parameters(self):
        rng = np.random.default_rng(self.seed)
        layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            if i < len(layer_sizes) - 2:
                scale = np.sqrt(2.0 / fan_in)
            else:
                scale = np.sqrt(1.0 / fan_in)
            self.weights.append(rng.normal(0.0, scale, size=(fan_in, fan_out)))
            self.biases.append(np.zeros((1, fan_out)))

    @staticmethod
    def relu(z):
        return np.maximum(0.0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)

    @staticmethod
    def softmax(logits):
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_values = np.exp(shifted)
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    @staticmethod
    def cross_entropy_loss(y_true_one_hot, probabilities):
        eps = 1e-12
        clipped = np.clip(probabilities, eps, 1.0 - eps)
        return float(-np.mean(np.sum(y_true_one_hot * np.log(clipped), axis=1)))

    def forward(self, X):
        activations = [X]
        pre_activations = []
        current = X

        for i in range(len(self.weights) - 1):
            z = current @ self.weights[i] + self.biases[i]
            current = self.relu(z)
            pre_activations.append(z)
            activations.append(current)

        logits = current @ self.weights[-1] + self.biases[-1]
        probabilities = self.softmax(logits)
        pre_activations.append(logits)
        activations.append(probabilities)
        return probabilities, activations, pre_activations

    def backward(self, y_true_one_hot, activations, pre_activations):
        batch_size = y_true_one_hot.shape[0]
        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)

        delta = (activations[-1] - y_true_one_hot) / batch_size
        grad_weights[-1] = activations[-2].T @ delta
        grad_biases[-1] = np.sum(delta, axis=0, keepdims=True)

        for i in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[i + 1].T) * self.relu_derivative(pre_activations[i])
            grad_weights[i] = activations[i].T @ delta
            grad_biases[i] = np.sum(delta, axis=0, keepdims=True)

        return grad_weights, grad_biases

    def update_parameters(self, grad_weights, grad_biases):
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_weights[i]
            self.biases[i] -= self.learning_rate * grad_biases[i]

    def train(
        self,
        X_train,
        y_train_one_hot,
        X_test,
        y_test_one_hot,
        epochs=500,
        batch_size=32,
        loss_threshold=0.02,
        patience=20,
        min_delta=1e-6,
        verbose=False,
    ):
        rng = np.random.default_rng(self.seed + 1000)
        history = {"train_loss": [], "test_loss": []}
        best_loss = np.inf
        no_improvement_count = 0
        stop_reason = "Đạt số epoch tối đa"
        start = time.time()
        epochs_completed = 0

        for epoch in range(1, epochs + 1):
            permutation = rng.permutation(X_train.shape[0])
            X_shuffled = X_train[permutation]
            y_shuffled = y_train_one_hot[permutation]

            for start_idx in range(0, X_train.shape[0], batch_size):
                end_idx = start_idx + batch_size
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                _, activations, pre_activations = self.forward(X_batch)
                grad_weights, grad_biases = self.backward(y_batch, activations, pre_activations)
                self.update_parameters(grad_weights, grad_biases)

            train_probabilities = self.predict_proba(X_train)
            test_probabilities = self.predict_proba(X_test)
            train_loss = self.cross_entropy_loss(y_train_one_hot, train_probabilities)
            test_loss = self.cross_entropy_loss(y_test_one_hot, test_probabilities)
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            epochs_completed = epoch

            if verbose and (epoch == 1 or epoch % 100 == 0):
                print(f"Epoch {epoch:4d}: train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")

            if train_loss < loss_threshold:
                stop_reason = f"Train loss < {loss_threshold}"
                break

            if train_loss < best_loss - min_delta:
                best_loss = train_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                stop_reason = f"Loss không cải thiện ít nhất {min_delta} sau {patience} epoch"
                break

        training_time = time.time() - start
        return history, epochs_completed, training_time, stop_reason

    def predict_proba(self, X):
        probabilities, _, _ = self.forward(X)
        return probabilities

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def evaluate_model(model, X, y, y_one_hot):
    probabilities = model.predict_proba(X)
    predictions = np.argmax(probabilities, axis=1)
    loss = model.cross_entropy_loss(y_one_hot, probabilities)
    accuracy = accuracy_score_manual(y, predictions)
    return loss, accuracy, predictions


def run_experiments(X_train, X_test, y_train, y_test, configs, num_classes):
    y_train_one_hot = one_hot_encode(y_train, num_classes)
    y_test_one_hot = one_hot_encode(y_test, num_classes)

    results = []
    histories = []

    for idx, config in enumerate(configs, start=1):
        model = MLPClassifierScratch(
            input_size=X_train.shape[1],
            hidden_layers=config["hidden_layers"],
            output_size=num_classes,
            learning_rate=config["learning_rate"],
            seed=config["seed"],
        )

        history, epochs_completed, training_time, stop_reason = model.train(
            X_train,
            y_train_one_hot,
            X_test,
            y_test_one_hot,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            loss_threshold=config["loss_threshold"],
            patience=config["patience"],
            min_delta=config.get("min_delta", 1e-6),
        )

        train_loss, train_acc, train_pred = evaluate_model(model, X_train, y_train, y_train_one_hot)
        test_loss, test_acc, test_pred = evaluate_model(model, X_test, y_test, y_test_one_hot)
        confusion = confusion_matrix_manual(y_test, test_pred, num_classes)

        result = {
            "STT": idx,
            "Cấu trúc mạng": str(config["hidden_layers"]),
            "Learning rate": config["learning_rate"],
            "Epoch tối đa": config["epochs"],
            "Điều kiện dừng": stop_reason,
            "Số bước lặp thực tế": epochs_completed,
            "Train loss": train_loss,
            "Test loss": test_loss,
            "Train accuracy": train_acc,
            "Test accuracy": test_acc,
            "Thời gian huấn luyện": training_time,
            "Nhận xét": "",
        }

        results.append(result)
        histories.append(
            {
                "config": config,
                "history": history,
                "model": model,
                "confusion_matrix": confusion,
                "test_predictions": test_pred,
                "train_predictions": train_pred,
            }
        )

        print(
            f"Config {idx}: hidden={config['hidden_layers']}, lr={config['learning_rate']}, "
            f"epochs={epochs_completed}, test_acc={test_acc:.4f}, test_loss={test_loss:.4f}, "
            f"time={training_time:.3f}s"
        )

    best_index = select_best_result(results)
    for i, result in enumerate(results):
        if i == best_index:
            result["Nhận xét"] = "Tốt nhất theo test accuracy; nếu bằng accuracy thì ưu tiên test loss thấp hơn"
        elif result["Train accuracy"] > result["Test accuracy"] + 0.20:
            result["Nhận xét"] = "Có dấu hiệu overfitting"
        elif result["Train accuracy"] < 0.45 and result["Test accuracy"] < 0.45:
            result["Nhận xét"] = "Có dấu hiệu underfitting"
        else:
            result["Nhận xét"] = "Kết quả trung bình"

    return results, histories, best_index


def select_best_result(results):
    sorted_indices = sorted(
        range(len(results)),
        key=lambda i: (-results[i]["Test accuracy"], results[i]["Test loss"], results[i]["Thời gian huấn luyện"]),
    )
    return sorted_indices[0]


def save_results_csv(results, output_path):
    fieldnames = list(results[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            formatted = row.copy()
            for key in ["Train loss", "Test loss", "Train accuracy", "Test accuracy", "Thời gian huấn luyện"]:
                formatted[key] = f"{row[key]:.6f}"
            writer.writerow(formatted)


def print_results_table(results):
    headers = [
        "STT",
        "Cấu trúc mạng",
        "LR",
        "Epoch max",
        "Dừng",
        "Epoch thực",
        "Train loss",
        "Test loss",
        "Train acc",
        "Test acc",
        "Time(s)",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result["STT"],
                result["Cấu trúc mạng"],
                result["Learning rate"],
                result["Epoch tối đa"],
                result["Điều kiện dừng"],
                result["Số bước lặp thực tế"],
                f"{result['Train loss']:.4f}",
                f"{result['Test loss']:.4f}",
                f"{result['Train accuracy']:.4f}",
                f"{result['Test accuracy']:.4f}",
                f"{result['Thời gian huấn luyện']:.3f}",
            ]
        )

    widths = [len(str(header)) for header in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))

    header_line = " | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers)))
    sep_line = "-+-".join("-" * width for width in widths)
    print("\nBẢNG SO SÁNH KẾT QUẢ")
    print(header_line)
    print(sep_line)
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))))


def plot_loss_curves(histories, best_index):
    plt.figure(figsize=(10, 6))
    for idx, item in enumerate(histories, start=1):
        train_loss = item["history"]["train_loss"]
        plt.plot(range(1, len(train_loss) + 1), train_loss, label=f"Config {idx}")
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title(f"Loss theo epoch cho {len(histories)} cấu hình MLP")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_ALL_PNG, dpi=160)
    plt.close()

    best_history = histories[best_index]["history"]
    epochs = range(1, len(best_history["train_loss"]) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, best_history["train_loss"], label="Train loss", linewidth=2)
    plt.plot(epochs, best_history["test_loss"], label="Test loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title(f"Loss của cấu hình tốt nhất: Config {best_index + 1}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_BEST_PNG, dpi=160)
    plt.close()


def markdown_table(results):
    headers = [
        "STT",
        "Cấu trúc mạng",
        "Learning rate",
        "Epoch tối đa",
        "Điều kiện dừng",
        "Số bước lặp thực tế",
        "Train loss",
        "Test loss",
        "Train accuracy",
        "Test accuracy",
        "Thời gian huấn luyện",
        "Nhận xét",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for result in results:
        row = [
            result["STT"],
            result["Cấu trúc mạng"],
            result["Learning rate"],
            result["Epoch tối đa"],
            result["Điều kiện dừng"],
            result["Số bước lặp thực tế"],
            f"{result['Train loss']:.4f}",
            f"{result['Test loss']:.4f}",
            f"{result['Train accuracy']:.4f}",
            f"{result['Test accuracy']:.4f}",
            f"{result['Thời gian huấn luyện']:.3f}s",
            result["Nhận xét"],
        ]
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def confusion_matrix_text(matrix):
    lines = ["Ma trận nhầm lẫn trên tập test (hàng là nhãn thật, cột là nhãn dự đoán):"]
    header = "      " + " ".join(f"{i:>4}" for i in range(matrix.shape[1]))
    lines.append(header)
    for i, row in enumerate(matrix):
        lines.append(f"{i:>4}: " + " ".join(f"{value:>4}" for value in row))
    return "\n".join(lines)


def build_report_text(df, feature_names, results, histories, best_index):
    best = results[best_index]
    best_history = histories[best_index]
    grade_counts = df["GRADE"].value_counts().sort_index()
    grade_distribution = ", ".join(f"{grade} {GRADE_NAMES[grade]}: {count}" for grade, count in grade_counts.items())
    feature_text = ", ".join(feature_names)

    report = f"""# Xây dựng mạng nơ-ron nhân tạo đa lớp MLP phân lớp Student Performance

**Môn học:** Học máy và Khai phá dữ liệu  
**Sinh viên:** Huy Nguyễn Đức  
**Mã sinh viên:** BIT220079  
**Ngày thực hiện:** {time.strftime('%d/%m/%Y')}

## 1. Giới thiệu bài toán

Bài toán đặt ra là dự đoán kết quả cuối kỳ của sinh viên dựa trên các thông tin cá nhân, gia đình, thói quen học tập và hoạt động học tập. Đây là bài toán phân lớp đa lớp vì nhãn đầu ra `GRADE` có 8 lớp: 0 Fail, 1 DD, 2 DC, 3 CC, 4 CB, 5 BB, 6 BA, 7 AA.

## 2. Mô tả dữ liệu

Dataset `Student performance.csv` gồm {df.shape[0]} dòng và {df.shape[1]} cột. Cột `STUDENT ID` là mã sinh viên, không dùng làm đặc trưng huấn luyện. Các cột `1` đến `30` là đặc trưng đầu vào, trong đó câu 1-10 mô tả thông tin cá nhân, câu 11-16 mô tả thông tin gia đình, các câu còn lại mô tả thói quen và hoạt động học tập. Cột `COURSE ID` được dùng như một đặc trưng đầu vào. Cột `GRADE` là nhãn cần dự đoán.

Phân bố nhãn trong toàn bộ dữ liệu: {grade_distribution}.

Các đặc trưng dùng huấn luyện: {feature_text}.

## 3. Tiền xử lý dữ liệu

Chương trình bỏ cột `STUDENT ID`, tách `X` là các đặc trưng và `y` là nhãn `GRADE`. Dữ liệu được chia train/test theo tỷ lệ 80/20 bằng hàm tự cài đặt dựa trên `numpy`, có giữ phân bố lớp theo từng nhãn. Sau khi chia, chương trình fit Min-Max Scaling trên tập train và dùng cùng tham số đó để chuẩn hóa tập test. Cách làm này tránh rò rỉ dữ liệu từ tập test vào quá trình huấn luyện. Nhãn `GRADE` được mã hóa one-hot bằng hàm tự cài đặt.

Chuẩn hóa dữ liệu là cần thiết vì các đặc trưng có thang giá trị khác nhau. Khi đưa các đặc trưng về cùng khoảng giá trị, quá trình Gradient Descent ổn định hơn, giảm nguy cơ một số đặc trưng có giá trị lớn chi phối gradient.

## 4. Cơ sở lý thuyết MLP

MLP là mạng nơ-ron truyền thẳng gồm lớp đầu vào, một hoặc nhiều lớp ẩn và lớp đầu ra. Ở mỗi lớp, dữ liệu được nhân với ma trận trọng số, cộng bias, sau đó đi qua hàm kích hoạt. Trong chương trình này, lớp ẩn dùng ReLU: `ReLU(z) = max(0, z)`. Lớp đầu ra dùng Softmax để chuyển logits thành xác suất cho 8 lớp.

Hàm mất mát dùng Cross Entropy cho phân lớp đa lớp. Với nhãn one-hot `y` và xác suất dự đoán `p`, loss được tính bằng trung bình của `-sum(y * log(p))` trên toàn bộ mẫu. Gradient được tính bằng backpropagation. Trọng số và bias được cập nhật bằng Mini-batch Gradient Descent.

## 5. Thiết kế chương trình

File `mlp_student_performance.py` gồm các phần chính:

- `load_data`: đọc dữ liệu CSV bằng pandas.
- `split_features_labels`: bỏ `STUDENT ID`, tách đặc trưng và nhãn.
- `stratified_train_test_split`: chia train/test bằng numpy, không dùng sklearn.
- `fit_minmax_scaler`, `transform_minmax`: chuẩn hóa Min-Max tự cài đặt.
- `one_hot_encode`: mã hóa nhãn one-hot tự cài đặt.
- `MLPClassifierScratch`: cài đặt MLP từ đầu, gồm khởi tạo trọng số, forward propagation, ReLU, Softmax, Cross Entropy, backpropagation và cập nhật Gradient Descent.
- `evaluate_model`: tính loss, accuracy và dự đoán.
- `run_experiments`: chạy nhiều cấu hình mạng.
- `plot_loss_curves`: vẽ biểu đồ loss.

Điều kiện dừng gồm: đạt số epoch tối đa, train loss nhỏ hơn ngưỡng cho trước, hoặc loss không cải thiện sau một số epoch `patience` với ngưỡng cải thiện tối thiểu `min_delta`. Các cấu hình thực nghiệm có thể thay đổi `patience` và `min_delta` để minh họa dừng sớm.

## 6. Thực nghiệm

Chương trình chạy {len(results)} cấu hình MLP khác nhau:

{markdown_table(results)}

Biểu đồ loss của cấu hình tốt nhất được lưu tại `{LOSS_BEST_PNG}`. Biểu đồ loss train của toàn bộ cấu hình được lưu tại `{LOSS_ALL_PNG}`.

Cấu hình tốt nhất là Config {best['STT']} với cấu trúc {best['Cấu trúc mạng']}, learning rate {best['Learning rate']}, test accuracy {best['Test accuracy']:.4f} và test loss {best['Test loss']:.4f}. Cấu hình này được chọn vì có test accuracy cao nhất; khi hòa accuracy thì ưu tiên test loss thấp hơn.

```text
{confusion_matrix_text(best_history['confusion_matrix'])}
```

## 7. Nhận xét và đánh giá

Kết quả cho thấy thay đổi số lớp ẩn, số neuron, learning rate và số epoch ảnh hưởng trực tiếp đến khả năng học của mô hình. Cấu hình quá nhỏ thường có khả năng biểu diễn hạn chế, dễ underfitting nếu cả train accuracy và test accuracy đều thấp. Cấu hình lớn hơn có thể học tốt hơn trên tập train, nhưng với dataset chỉ có 145 mẫu, nếu chênh lệch train accuracy và test accuracy quá cao thì có dấu hiệu overfitting.

Learning rate lớn giúp mô hình học nhanh hơn nhưng có thể làm loss dao động. Learning rate nhỏ ổn định hơn nhưng cần nhiều epoch hơn. Điều kiện dừng sớm giúp tránh huấn luyện không cần thiết khi loss không còn cải thiện rõ rệt.

## 8. Kết luận

Bài làm đã cài đặt đầy đủ MLP từ đầu bằng numpy, không dùng sklearn, tensorflow, keras, pytorch hoặc classifier có sẵn. Chương trình đọc dữ liệu, tiền xử lý, chia train/test, huấn luyện {len(results)} cấu hình, đánh giá bằng loss và accuracy, xuất bảng kết quả, vẽ biểu đồ loss và tạo báo cáo.

Hạn chế chính là dataset nhỏ, nhiều đặc trưng dạng categorical được mã số, nên mô hình có thể chưa tổng quát tốt. Các cải tiến có thể thử gồm k-fold cross validation, điều chỉnh learning rate, thêm regularization, thử one-hot cho các đặc trưng categorical và mở rộng tìm kiếm kiến trúc mạng.
"""
    return report


def write_report_markdown(report_text, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)


def add_text_page(pdf, title, lines, fontsize=10):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    y = 0.96
    fig.text(0.08, y, title, fontsize=15, weight="bold", va="top")
    y -= 0.05

    for line in lines:
        if y < 0.06:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            y = 0.96

        if line.strip() == "":
            y -= 0.018
            continue

        wrapped = textwrap.wrap(line, width=96, replace_whitespace=False) or [""]
        for subline in wrapped:
            if y < 0.06:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.patch.set_facecolor("white")
                y = 0.96
            fig.text(0.08, y, subline, fontsize=fontsize, va="top")
            y -= 0.022

    pdf.savefig(fig)
    plt.close(fig)


def write_report_pdf(report_text, results, best_index, output_path):
    with PdfPages(output_path) as pdf:
        cleaned_lines = []
        for line in report_text.splitlines():
            if line.startswith("# "):
                cleaned_lines.append(line.replace("# ", ""))
            elif line.startswith("## "):
                cleaned_lines.append("")
                cleaned_lines.append(line.replace("## ", ""))
            elif line.startswith("|") or line.startswith("```"):
                continue
            elif line.startswith("- "):
                cleaned_lines.append("• " + line[2:])
            else:
                cleaned_lines.append(line.replace("**", "").replace("`", ""))

        add_text_page(pdf, "Báo cáo MLP Student Performance", cleaned_lines, fontsize=9.5)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        headers = [
            "STT",
            "Cấu trúc",
            "LR",
            "Epoch",
            "Epoch thực",
            "Train loss",
            "Test loss",
            "Train acc",
            "Test acc",
            "Time(s)",
        ]
        rows = []
        for result in results:
            rows.append(
                [
                    result["STT"],
                    result["Cấu trúc mạng"],
                    result["Learning rate"],
                    result["Epoch tối đa"],
                    result["Số bước lặp thực tế"],
                    f"{result['Train loss']:.4f}",
                    f"{result['Test loss']:.4f}",
                    f"{result['Train accuracy']:.4f}",
                    f"{result['Test accuracy']:.4f}",
                    f"{result['Thời gian huấn luyện']:.3f}",
                ]
            )
        table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        ax.set_title(f"Bảng so sánh kết quả - cấu hình tốt nhất: Config {best_index + 1}", fontsize=14, pad=20)
        pdf.savefig(fig)
        plt.close(fig)

        for image_path in [LOSS_BEST_PNG, LOSS_ALL_PNG]:
            if Path(image_path).exists():
                fig = plt.figure(figsize=(11.69, 8.27))
                image = plt.imread(image_path)
                plt.imshow(image)
                plt.axis("off")
                pdf.savefig(fig)
                plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / DATA_FILE

    df = load_data(csv_path)
    X, y, feature_names = split_features_labels(df)
    num_classes = len(np.unique(y))

    X_train_raw, X_test_raw, y_train, y_test = stratified_train_test_split(X, y, test_size=0.2, seed=42)
    min_values, ranges = fit_minmax_scaler(X_train_raw)
    X_train = transform_minmax(X_train_raw, min_values, ranges)
    X_test = transform_minmax(X_test_raw, min_values, ranges)

    configs = [
        {"hidden_layers": [16], "learning_rate": 0.01, "epochs": 500, "batch_size": 32, "loss_threshold": 0.02, "patience": 20, "min_delta": 1e-6, "seed": 11},
        {"hidden_layers": [32], "learning_rate": 0.01, "epochs": 500, "batch_size": 32, "loss_threshold": 0.02, "patience": 20, "min_delta": 1e-6, "seed": 12},
        {"hidden_layers": [32, 16], "learning_rate": 0.01, "epochs": 1000, "batch_size": 32, "loss_threshold": 0.02, "patience": 20, "min_delta": 1e-6, "seed": 13},
        {"hidden_layers": [64, 32], "learning_rate": 0.005, "epochs": 1000, "batch_size": 32, "loss_threshold": 0.02, "patience": 20, "min_delta": 1e-6, "seed": 14},
        {"hidden_layers": [64, 32, 16], "learning_rate": 0.001, "epochs": 1500, "batch_size": 32, "loss_threshold": 0.02, "patience": 20, "min_delta": 1e-6, "seed": 15},
        {"hidden_layers": [16], "learning_rate": 0.01, "epochs": 1000, "batch_size": 32, "loss_threshold": 1.60, "patience": 20, "min_delta": 1e-6, "seed": 16},
        {"hidden_layers": [16], "learning_rate": 0.0001, "epochs": 500, "batch_size": 32, "loss_threshold": 0.02, "patience": 10, "min_delta": 1e-2, "seed": 17},
    ]

    print(f"Dataset: {df.shape[0]} mẫu, {df.shape[1]} cột")
    print(f"Train: {X_train.shape[0]} mẫu, Test: {X_test.shape[0]} mẫu, Số đặc trưng: {X_train.shape[1]}")
    print(f"Bắt đầu huấn luyện {len(configs)} cấu hình MLP tự cài đặt...\n")

    results, histories, best_index = run_experiments(X_train, X_test, y_train, y_test, configs, num_classes)
    print_results_table(results)

    save_results_csv(results, base_dir / RESULTS_CSV)
    plot_loss_curves(histories, best_index)
    report_text = build_report_text(df, feature_names, results, histories, best_index)
    write_report_markdown(report_text, base_dir / REPORT_MD)
    write_report_pdf(report_text, results, best_index, base_dir / REPORT_PDF)

    best = results[best_index]
    print("\nCấu hình tốt nhất:")
    print(
        f"Config {best['STT']} - hidden_layers={best['Cấu trúc mạng']}, "
        f"test_accuracy={best['Test accuracy']:.4f}, test_loss={best['Test loss']:.4f}"
    )
    print("\nĐã tạo các file:")
    print(f"- {RESULTS_CSV}")
    print(f"- {LOSS_BEST_PNG}")
    print(f"- {LOSS_ALL_PNG}")
    print(f"- {REPORT_MD}")
    print(f"- {REPORT_PDF}")


if __name__ == "__main__":
    main()
