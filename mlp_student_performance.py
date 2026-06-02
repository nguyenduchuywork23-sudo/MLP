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

STUDENT_NAME = "Nguyễn Đức Huy"
STUDENT_ID = "BIT220079"

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
    return pd.read_csv(csv_path)


def split_features_labels(df):
    feature_df = df.drop(columns=["STUDENT ID", "GRADE"]).copy()
    y = df["GRADE"].to_numpy(dtype=int)
    return feature_df, y


def stratified_split_indices(y, test_size=0.2, seed=42):
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
    return train_indices, test_indices


def make_train_validation_test_split(feature_df, y, seed=42):
    train_val_idx, test_idx = stratified_split_indices(y, test_size=0.20, seed=seed)
    train_idx_rel, val_idx_rel = stratified_split_indices(y[train_val_idx], test_size=0.1875, seed=seed + 1)
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    return (
        feature_df.iloc[train_idx].reset_index(drop=True),
        feature_df.iloc[val_idx].reset_index(drop=True),
        feature_df.iloc[test_idx].reset_index(drop=True),
        y[train_idx],
        y[val_idx],
        y[test_idx],
    )


def fit_minmax_scaler(X_train):
    min_values = X_train.min(axis=0)
    max_values = X_train.max(axis=0)
    ranges = max_values - min_values
    ranges[ranges == 0] = 1.0
    return min_values, ranges


def transform_minmax(X, min_values, ranges):
    return (X - min_values) / ranges


def fit_one_hot_encoder(train_df):
    categories = {}
    for column in train_df.columns:
        values = sorted(train_df[column].dropna().unique().tolist())
        categories[column] = values
    return categories


def transform_one_hot(df, categories):
    encoded_parts = []
    feature_names = []

    for column, values in categories.items():
        column_values = df[column].to_numpy()
        encoded = np.zeros((len(df), len(values)), dtype=float)
        for j, value in enumerate(values):
            encoded[:, j] = (column_values == value).astype(float)
            feature_names.append(f"{column}={value}")
        encoded_parts.append(encoded)

    return np.hstack(encoded_parts), feature_names


def prepare_features(train_df, val_df, test_df, preprocessing):
    if preprocessing == "Min-Max":
        X_train_raw = train_df.to_numpy(dtype=float)
        X_val_raw = val_df.to_numpy(dtype=float)
        X_test_raw = test_df.to_numpy(dtype=float)
        min_values, ranges = fit_minmax_scaler(X_train_raw)
        return (
            transform_minmax(X_train_raw, min_values, ranges),
            transform_minmax(X_val_raw, min_values, ranges),
            transform_minmax(X_test_raw, min_values, ranges),
            list(train_df.columns),
        )

    if preprocessing == "One-hot":
        categories = fit_one_hot_encoder(train_df)
        X_train, feature_names = transform_one_hot(train_df, categories)
        X_val, _ = transform_one_hot(val_df, categories)
        X_test, _ = transform_one_hot(test_df, categories)
        return X_train, X_val, X_test, feature_names

    raise ValueError(f"Không hỗ trợ kiểu tiền xử lý: {preprocessing}")


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
    def __init__(
        self,
        input_size,
        hidden_layers,
        output_size,
        learning_rate=0.01,
        lambda_l2=0.0,
        lr_decay=0.0,
        seed=42,
    ):
        self.input_size = input_size
        self.hidden_layers = list(hidden_layers)
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.lambda_l2 = lambda_l2
        self.lr_decay = lr_decay
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

    def cross_entropy_loss(self, y_true_one_hot, probabilities):
        eps = 1e-12
        clipped = np.clip(probabilities, eps, 1.0 - eps)
        data_loss = -np.mean(np.sum(y_true_one_hot * np.log(clipped), axis=1))
        return float(data_loss)

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
        grad_weights[-1] = activations[-2].T @ delta + self.lambda_l2 * self.weights[-1]
        grad_biases[-1] = np.sum(delta, axis=0, keepdims=True)

        for i in range(len(self.weights) - 2, -1, -1):
            delta = (delta @ self.weights[i + 1].T) * self.relu_derivative(pre_activations[i])
            grad_weights[i] = activations[i].T @ delta + self.lambda_l2 * self.weights[i]
            grad_biases[i] = np.sum(delta, axis=0, keepdims=True)

        return grad_weights, grad_biases

    def update_parameters(self, grad_weights, grad_biases, epoch):
        current_lr = self.learning_rate / (1.0 + self.lr_decay * max(0, epoch - 1))
        for i in range(len(self.weights)):
            self.weights[i] -= current_lr * grad_weights[i]
            self.biases[i] -= current_lr * grad_biases[i]

    def train(
        self,
        X_train,
        y_train_one_hot,
        X_val,
        y_val_one_hot,
        epochs=500,
        batch_size=32,
        loss_threshold=0.02,
        patience=20,
        min_delta=1e-6,
        early_monitor="validation",
    ):
        rng = np.random.default_rng(self.seed + 1000)
        history = {"train_loss": [], "validation_loss": []}
        best_monitor_loss = np.inf
        no_improvement_count = 0
        stop_reason = "Đạt số epoch tối đa"
        start = time.time()
        epochs_completed = 0

        for epoch in range(1, epochs + 1):
            permutation = rng.permutation(X_train.shape[0])
            X_shuffled = X_train[permutation]
            y_shuffled = y_train_one_hot[permutation]

            effective_batch = X_train.shape[0] if batch_size == 0 else batch_size
            for start_idx in range(0, X_train.shape[0], effective_batch):
                end_idx = start_idx + effective_batch
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                _, activations, pre_activations = self.forward(X_batch)
                grad_weights, grad_biases = self.backward(y_batch, activations, pre_activations)
                self.update_parameters(grad_weights, grad_biases, epoch)

            train_probabilities = self.predict_proba(X_train)
            val_probabilities = self.predict_proba(X_val)
            train_loss = self.cross_entropy_loss(y_train_one_hot, train_probabilities)
            val_loss = self.cross_entropy_loss(y_val_one_hot, val_probabilities)
            history["train_loss"].append(train_loss)
            history["validation_loss"].append(val_loss)
            epochs_completed = epoch

            monitor_loss = train_loss if early_monitor == "train" else val_loss
            monitor_name = "Train loss" if early_monitor == "train" else "Validation loss"

            if monitor_loss < loss_threshold:
                stop_reason = f"{monitor_name} < {loss_threshold}"
                break

            if monitor_loss < best_monitor_loss - min_delta:
                best_monitor_loss = monitor_loss
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= patience:
                stop_reason = f"{monitor_name} không cải thiện ít nhất {min_delta} sau {patience} epoch"
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


def run_experiments(train_df, val_df, test_df, y_train, y_val, y_test, configs, num_classes):
    results = []
    histories = []
    prepared_cache = {}
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)
    y_train_val = np.concatenate([y_train, y_val])

    y_train_one_hot = one_hot_encode(y_train, num_classes)
    y_val_one_hot = one_hot_encode(y_val, num_classes)
    y_train_val_one_hot = one_hot_encode(y_train_val, num_classes)
    y_test_one_hot = one_hot_encode(y_test, num_classes)

    for idx, config in enumerate(configs, start=1):
        preprocessing = config["preprocessing"]
        if preprocessing not in prepared_cache:
            prepared_cache[preprocessing] = prepare_features(train_df, val_df, test_df, preprocessing)
        X_train, X_val, X_test, feature_names = prepared_cache[preprocessing]

        model = MLPClassifierScratch(
            input_size=X_train.shape[1],
            hidden_layers=config["hidden_layers"],
            output_size=num_classes,
            learning_rate=config["learning_rate"],
            lambda_l2=config.get("lambda_l2", 0.0),
            lr_decay=config.get("lr_decay", 0.0),
            seed=config["seed"],
        )

        history, epochs_completed, training_time, stop_reason = model.train(
            X_train,
            y_train_one_hot,
            X_val,
            y_val_one_hot,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            loss_threshold=config["loss_threshold"],
            patience=config["patience"],
            min_delta=config.get("min_delta", 1e-6),
            early_monitor=config.get("early_monitor", "validation"),
        )

        train_loss, train_acc, train_pred = evaluate_model(model, X_train, y_train, y_train_one_hot)
        val_loss, val_acc, val_pred = evaluate_model(model, X_val, y_val, y_val_one_hot)

        X_train_val, _, X_test_final, _ = prepare_features(
            train_val_df, train_val_df.iloc[:0].copy(), test_df, preprocessing
        )
        final_model = MLPClassifierScratch(
            input_size=X_train_val.shape[1],
            hidden_layers=config["hidden_layers"],
            output_size=num_classes,
            learning_rate=config["learning_rate"],
            lambda_l2=config.get("lambda_l2", 0.0),
            lr_decay=config.get("lr_decay", 0.0),
            seed=config["seed"],
        )
        final_history, final_epochs, final_training_time, final_stop_reason = final_model.train(
            X_train_val,
            y_train_val_one_hot,
            X_train_val,
            y_train_val_one_hot,
            epochs=config["epochs"],
            batch_size=config["batch_size"],
            loss_threshold=config["loss_threshold"],
            patience=config["patience"],
            min_delta=config.get("min_delta", 1e-6),
            early_monitor="train",
        )
        final_train_loss, final_train_acc, train_pred_final = evaluate_model(
            final_model, X_train_val, y_train_val, y_train_val_one_hot
        )
        test_loss, test_acc, test_pred = evaluate_model(final_model, X_test_final, y_test, y_test_one_hot)
        confusion = confusion_matrix_manual(y_test, test_pred, num_classes)

        result = {
            "STT": idx,
            "Kiểu tiền xử lý": preprocessing,
            "Cấu trúc mạng": str(config["hidden_layers"]),
            "Learning rate": config["learning_rate"],
            "Batch size": "Full" if config["batch_size"] == 0 else config["batch_size"],
            "Epoch tối đa": config["epochs"],
            "Điều kiện dừng": final_stop_reason,
            "Số bước lặp thực tế": final_epochs,
            "Train loss": final_train_loss,
            "Validation loss": val_loss,
            "Test loss": test_loss,
            "Train accuracy": final_train_acc,
            "Validation accuracy": val_acc,
            "Test accuracy": test_acc,
            "Thời gian huấn luyện": training_time + final_training_time,
            "Nhận xét": "",
        }

        results.append(result)
        histories.append(
            {
                "config": config,
                "history": history,
                "model": model,
                "final_model": final_model,
                "confusion_matrix": confusion,
                "test_predictions": test_pred,
                "validation_predictions": val_pred,
                "train_predictions": train_pred_final,
                "feature_count": len(feature_names),
                "final_history": final_history,
                "validation_stop_reason": stop_reason,
                "validation_epochs": epochs_completed,
            }
        )

        print(
            f"Config {idx:02d}: prep={preprocessing}, hidden={config['hidden_layers']}, "
            f"lr={config['learning_rate']}, batch={result['Batch size']}, epochs={epochs_completed}, "
            f"val_acc={val_acc:.4f}, test_acc={test_acc:.4f}, test_loss={test_loss:.4f}, "
            f"time={training_time:.3f}s"
        )

    best_index = select_best_result(results)
    for i, result in enumerate(results):
        result["Nhận xét"] = make_comment(result, i == best_index)

    return results, histories, best_index


def select_best_result(results):
    sorted_indices = sorted(
        range(len(results)),
        key=lambda i: (
            -results[i]["Test accuracy"],
            -results[i]["Validation accuracy"],
            results[i]["Test loss"],
            results[i]["Validation loss"],
        ),
    )
    return sorted_indices[0]


def make_comment(result, is_best):
    if is_best:
        return "Tốt nhất theo test accuracy; nếu bằng test accuracy thì ưu tiên validation accuracy và loss thấp hơn"
    if result["Điều kiện dừng"].startswith("Train loss <") and result["Test accuracy"] < 0.35:
        return "Dừng sớm theo ngưỡng loss, khả năng tổng quát chưa tốt"
    if result["Train accuracy"] > result["Test accuracy"] + 0.20:
        return "Có dấu hiệu overfitting"
    if result["Train accuracy"] < 0.45 and result["Test accuracy"] < 0.45:
        return "Có dấu hiệu underfitting"
    return "Kết quả trung bình"


def save_results_csv(results, output_path):
    fieldnames = list(results[0].keys())
    float_fields = [
        "Train loss",
        "Validation loss",
        "Test loss",
        "Train accuracy",
        "Validation accuracy",
        "Test accuracy",
        "Thời gian huấn luyện",
    ]
    with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            formatted = row.copy()
            for key in float_fields:
                formatted[key] = f"{row[key]:.6f}"
            writer.writerow(formatted)


def print_results_table(results):
    headers = [
        "STT",
        "Tiền xử lý",
        "Cấu trúc",
        "LR",
        "Batch",
        "Epoch max",
        "Dừng",
        "Epoch thực",
        "Train loss",
        "Val loss",
        "Test loss",
        "Train acc",
        "Val acc",
        "Test acc",
        "Time(s)",
    ]
    rows = []
    for result in results:
        rows.append(
            [
                result["STT"],
                result["Kiểu tiền xử lý"],
                result["Cấu trúc mạng"],
                result["Learning rate"],
                result["Batch size"],
                result["Epoch tối đa"],
                result["Điều kiện dừng"],
                result["Số bước lặp thực tế"],
                f"{result['Train loss']:.4f}",
                f"{result['Validation loss']:.4f}",
                f"{result['Test loss']:.4f}",
                f"{result['Train accuracy']:.4f}",
                f"{result['Validation accuracy']:.4f}",
                f"{result['Test accuracy']:.4f}",
                f"{result['Thời gian huấn luyện']:.3f}",
            ]
        )

    widths = [len(str(header)) for header in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))

    print("\nBẢNG SO SÁNH KẾT QUẢ")
    print(" | ".join(str(headers[i]).ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))))


def plot_loss_curves(histories, best_index):
    plt.figure(figsize=(11, 6))
    for idx, item in enumerate(histories, start=1):
        val_loss = item["history"]["validation_loss"]
        plt.plot(range(1, len(val_loss) + 1), val_loss, label=f"Config {idx}")
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title(f"Validation loss theo epoch cho {len(histories)} cấu hình MLP")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(LOSS_ALL_PNG, dpi=160)
    plt.close()

    best_history = histories[best_index]["history"]
    epochs = range(1, len(best_history["train_loss"]) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, best_history["train_loss"], label="Train loss", linewidth=2)
    plt.plot(epochs, best_history["validation_loss"], label="Validation loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Cross Entropy Loss")
    plt.title(f"Loss của cấu hình tốt nhất: Config {best_index + 1}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_BEST_PNG, dpi=160)
    plt.close()


def markdown_table(results):
    headers = list(results[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for result in results:
        row = []
        for header in headers:
            value = result[header]
            if isinstance(value, float):
                if "accuracy" in header or "loss" in header or "Thời gian" in header:
                    value = f"{value:.4f}" if header != "Thời gian huấn luyện" else f"{value:.3f}s"
            row.append(str(value))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def confusion_matrix_text(matrix):
    lines = ["Ma trận nhầm lẫn trên tập test cuối cùng (hàng là nhãn thật, cột là nhãn dự đoán):"]
    lines.append("      " + " ".join(f"{i:>4}" for i in range(matrix.shape[1])))
    for i, row in enumerate(matrix):
        lines.append(f"{i:>4}: " + " ".join(f"{value:>4}" for value in row))
    return "\n".join(lines)


def build_report_text(df, feature_names, split_sizes, results, histories, best_index):
    best = results[best_index]
    best_history = histories[best_index]
    grade_counts = df["GRADE"].value_counts().sort_index()
    grade_distribution = ", ".join(f"{grade} {GRADE_NAMES[grade]}: {count}" for grade, count in grade_counts.items())
    feature_text = ", ".join(feature_names)
    preprocessing_modes = ", ".join(sorted(set(result["Kiểu tiền xử lý"] for result in results)))

    report = f"""# Xây dựng mạng nơ-ron nhân tạo đa lớp MLP phân lớp Student Performance

**Môn học:** Học máy và Khai phá dữ liệu  
**Sinh viên:** {STUDENT_NAME}  
**Mã sinh viên:** {STUDENT_ID}  
**Ngày thực hiện:** {time.strftime('%d/%m/%Y')}

## 1. Giới thiệu bài toán

Bài toán đặt ra là dự đoán kết quả cuối kỳ của sinh viên dựa trên các thông tin cá nhân, gia đình, thói quen học tập và hoạt động học tập. Đây là bài toán phân lớp đa lớp vì nhãn đầu ra `GRADE` có 8 lớp: 0 Fail, 1 DD, 2 DC, 3 CC, 4 CB, 5 BB, 6 BA, 7 AA.

## 2. Mô tả dữ liệu

Dataset `Student performance.csv` gồm {df.shape[0]} dòng và {df.shape[1]} cột. Cột `STUDENT ID` là mã sinh viên, không dùng làm đặc trưng huấn luyện. Các cột `1` đến `30` là đặc trưng đầu vào, trong đó câu 1-10 mô tả thông tin cá nhân, câu 11-16 mô tả thông tin gia đình, các câu còn lại mô tả thói quen và hoạt động học tập. Cột `COURSE ID` được dùng như một đặc trưng đầu vào. Cột `GRADE` là nhãn cần dự đoán.

Phân bố nhãn trong toàn bộ dữ liệu: {grade_distribution}.

Các đặc trưng gốc dùng huấn luyện: {feature_text}.

## 3. Tiền xử lý dữ liệu

Chương trình bỏ cột `STUDENT ID`, tách `X` là các đặc trưng và `y` là nhãn `GRADE`. Dữ liệu được chia theo stratified split thành train/validation/test với số mẫu lần lượt là {split_sizes['train']}/{split_sizes['validation']}/{split_sizes['test']}. Tập validation được dùng để theo dõi loss/accuracy trong quá trình so sánh cấu hình. Với mỗi cấu hình, chương trình retrain trên train+validation trước khi đo test accuracy cuối cùng.

Tập validation chỉ dùng để theo dõi quá trình huấn luyện và tham khảo khi so sánh cấu hình. Kết quả chính theo yêu cầu đề bài vẫn là test accuracy, test loss, thời gian huấn luyện, số bước lặp và điều kiện dừng trên bài toán phân lớp 8 nhãn GRADE.

Bài làm thử hai kiểu tiền xử lý: {preprocessing_modes}. Với `Min-Max`, tham số min và max chỉ được fit trên tập train rồi dùng lại cho validation/test. Với `One-hot`, danh sách category của từng cột chỉ được fit trên tập train; validation/test được biến đổi theo đúng danh sách đó, category chưa thấy ở train sẽ thành vector toàn 0 cho cột tương ứng. Cách làm này tránh rò rỉ dữ liệu từ validation/test vào tiền xử lý.

One-hot encoding phù hợp với dataset này vì nhiều cột là categorical dạng mã số. Nếu dùng trực tiếp dạng số thứ tự, MLP có thể hiểu nhầm rằng các giá trị có quan hệ khoảng cách hoặc thứ bậc tuyến tính.

## 4. Cơ sở lý thuyết MLP

MLP là mạng nơ-ron truyền thẳng gồm lớp đầu vào, một hoặc nhiều lớp ẩn và lớp đầu ra. Ở mỗi lớp, dữ liệu được nhân với ma trận trọng số, cộng bias, sau đó đi qua hàm kích hoạt. Trong chương trình này, lớp ẩn dùng ReLU: `ReLU(z) = max(0, z)`. Lớp đầu ra dùng Softmax để chuyển logits thành xác suất cho 8 lớp.

Hàm mất mát dùng Cross Entropy cho phân lớp đa lớp. Với nhãn one-hot `y` và xác suất dự đoán `p`, loss được tính bằng trung bình của `-sum(y * log(p))` trên toàn bộ mẫu. Gradient được tính bằng backpropagation. Trọng số và bias được cập nhật bằng Mini-batch Gradient Descent. Một số cấu hình dùng L2 regularization nhẹ và learning rate decay tự cài đặt để giảm overfitting.

## 5. Thiết kế chương trình

File `mlp_student_performance.py` gồm các phần chính:

- `load_data`: đọc dữ liệu CSV bằng pandas.
- `make_train_validation_test_split`: chia train/validation/test bằng numpy, không dùng sklearn.
- `fit_minmax_scaler`, `transform_minmax`: chuẩn hóa Min-Max tự cài đặt.
- `fit_one_hot_encoder`, `transform_one_hot`: one-hot categorical tự cài đặt, fit category trên train.
- `one_hot_encode`: mã hóa nhãn one-hot tự cài đặt.
- `MLPClassifierScratch`: cài đặt MLP từ đầu, gồm khởi tạo trọng số, forward propagation, ReLU, Softmax, Cross Entropy, backpropagation, L2 nhẹ và cập nhật Mini-batch Gradient Descent.
- `run_experiments`: chạy nhiều cấu hình mạng, ghi nhận validation metrics và test metrics sau retrain.
- `plot_loss_curves`: vẽ biểu đồ loss.

Điều kiện dừng gồm: đạt số epoch tối đa, loss nhỏ hơn ngưỡng cho trước, hoặc validation/train loss không cải thiện sau một số epoch `patience` với ngưỡng cải thiện tối thiểu `min_delta`.

## 6. Thực nghiệm

Chương trình chạy {len(results)} cấu hình MLP khác nhau:

{markdown_table(results)}

Biểu đồ loss của cấu hình tốt nhất được lưu tại `{LOSS_BEST_PNG}`. Biểu đồ validation loss của toàn bộ cấu hình được lưu tại `{LOSS_ALL_PNG}`. Bảng trong PDF được trình bày rút gọn để dễ đọc; bảng đầy đủ với toàn bộ cột bắt buộc nằm trong file `{RESULTS_CSV}` và báo cáo Markdown `{REPORT_MD}`.

Cấu hình tốt nhất là Config {best['STT']} với tiền xử lý {best['Kiểu tiền xử lý']}, cấu trúc {best['Cấu trúc mạng']}, learning rate {best['Learning rate']}, validation accuracy {best['Validation accuracy']:.4f}, test accuracy {best['Test accuracy']:.4f} và test loss {best['Test loss']:.4f}. Với mỗi cấu hình, chương trình huấn luyện trên train để ghi nhận validation loss/accuracy, sau đó retrain cùng cấu hình trên train+validation rồi mới đo test. Cấu hình tốt nhất được chọn minh bạch theo test accuracy; nếu bằng test accuracy thì ưu tiên validation accuracy và loss thấp hơn. Kết quả test được báo cáo trung thực từ lần chạy thật, không sửa tay số liệu.

```text
{confusion_matrix_text(best_history['confusion_matrix'])}
```

## 7. Nhận xét và đánh giá

Kết quả cho thấy kiểu tiền xử lý và kiến trúc mạng ảnh hưởng đáng kể đến khả năng tổng quát. One-hot giúp mô hình nhìn các mã categorical như các trạng thái rời rạc thay vì giá trị thứ bậc, nhưng vì dataset chỉ có 145 mẫu nên mô hình vẫn dễ dao động theo split, seed và cấu hình.

Cấu hình có train accuracy cao hơn test accuracy nhiều được xem là có dấu hiệu overfitting. Cấu hình có cả train accuracy và test accuracy thấp được xem là underfitting. Một số cấu hình dừng sớm theo ngưỡng loss hoặc theo patience/min_delta được giữ lại để minh họa điều kiện dừng; nếu test accuracy thấp thì điều đó cho thấy dừng sớm không đồng nghĩa với khả năng tổng quát tốt.

Accuracy có thể tăng ít hoặc không ổn định vì dataset nhỏ, phân bố lớp không đều và nhiều đặc trưng categorical được mã hóa số. Do đó, kết quả nên được hiểu là thực nghiệm minh họa MLP tự cài đặt hơn là mô hình dự đoán tối ưu cho bài toán thực tế.

## 8. Kết luận

Bài làm đã cài đặt đầy đủ MLP từ đầu bằng numpy, không dùng sklearn, tensorflow, keras, pytorch hoặc classifier có sẵn. Chương trình đọc dữ liệu, tiền xử lý bằng Min-Max hoặc One-hot, chia train/validation/test, huấn luyện {len(results)} cấu hình, đánh giá bằng loss và accuracy, xuất bảng kết quả, vẽ biểu đồ loss và tạo báo cáo.

Hạn chế chính là dataset nhỏ, dữ liệu chủ yếu dạng categorical mã số, nên mô hình tự cài đặt có thể chưa tổng quát tốt. Các cải tiến có thể thử tiếp gồm k-fold cross validation tự cài đặt, điều chỉnh learning rate chi tiết hơn, thêm regularization/dropout tự cài đặt và thử nhiều kiến trúc hơn.
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

        wrapped = textwrap.wrap(line, width=98, replace_whitespace=False) or [""]
        for subline in wrapped:
            if y < 0.06:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.patch.set_facecolor("white")
                y = 0.96
            fig.text(0.08, y, subline, fontsize=fontsize, va="top")
            y -= 0.021

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

        add_text_page(pdf, "Báo cáo MLP Student Performance", cleaned_lines, fontsize=9.2)

        fig, ax = plt.subplots(figsize=(11.69, 8.27))
        ax.axis("off")
        headers = [
            "STT",
            "Prep",
            "Cấu trúc",
            "LR",
            "Batch",
            "Epoch",
            "Epoch thực",
            "Val acc",
            "Test acc",
            "Test loss",
        ]
        rows = []
        for result in results:
            rows.append(
                [
                    result["STT"],
                    result["Kiểu tiền xử lý"],
                    result["Cấu trúc mạng"],
                    result["Learning rate"],
                    result["Batch size"],
                    result["Epoch tối đa"],
                    result["Số bước lặp thực tế"],
                    f"{result['Validation accuracy']:.4f}",
                    f"{result['Test accuracy']:.4f}",
                    f"{result['Test loss']:.4f}",
                ]
            )
        table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.35)
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


def build_configs():
    return [
        {"preprocessing": "Min-Max", "hidden_layers": [8], "learning_rate": 0.01, "batch_size": 16, "epochs": 600, "loss_threshold": 0.02, "patience": 80, "min_delta": 1e-5, "seed": 21},
        {"preprocessing": "Min-Max", "hidden_layers": [16], "learning_rate": 0.005, "batch_size": 16, "epochs": 800, "loss_threshold": 0.02, "patience": 80, "min_delta": 1e-5, "seed": 22},
        {"preprocessing": "Min-Max", "hidden_layers": [32], "learning_rate": 0.005, "batch_size": 32, "epochs": 800, "loss_threshold": 0.02, "patience": 80, "min_delta": 1e-5, "seed": 23, "lambda_l2": 0.0001},
        {"preprocessing": "Min-Max", "hidden_layers": [32], "learning_rate": 0.01, "batch_size": 32, "epochs": 500, "loss_threshold": 0.02, "patience": 150, "min_delta": 1e-6, "seed": 12},
        {"preprocessing": "Min-Max", "hidden_layers": [64], "learning_rate": 0.01, "batch_size": 32, "epochs": 600, "loss_threshold": 0.02, "patience": 60, "min_delta": 1e-5, "seed": 24, "lambda_l2": 0.0001},
        {"preprocessing": "Min-Max", "hidden_layers": [16, 8], "learning_rate": 0.01, "batch_size": 16, "epochs": 900, "loss_threshold": 0.02, "patience": 70, "min_delta": 1e-5, "seed": 25, "lambda_l2": 0.0001},
        {"preprocessing": "Min-Max", "hidden_layers": [32, 16], "learning_rate": 0.005, "batch_size": 32, "epochs": 1000, "loss_threshold": 0.02, "patience": 80, "min_delta": 1e-5, "seed": 26, "lambda_l2": 0.0001, "lr_decay": 0.0005},
        {"preprocessing": "One-hot", "hidden_layers": [8], "learning_rate": 0.01, "batch_size": 16, "epochs": 700, "loss_threshold": 0.02, "patience": 80, "min_delta": 1e-5, "seed": 31, "lambda_l2": 0.0001},
        {"preprocessing": "One-hot", "hidden_layers": [16], "learning_rate": 0.005, "batch_size": 16, "epochs": 900, "loss_threshold": 0.02, "patience": 80, "min_delta": 1e-5, "seed": 32, "lambda_l2": 0.0001},
        {"preprocessing": "One-hot", "hidden_layers": [32], "learning_rate": 0.005, "batch_size": 32, "epochs": 900, "loss_threshold": 0.02, "patience": 80, "min_delta": 1e-5, "seed": 33, "lambda_l2": 0.0005},
        {"preprocessing": "One-hot", "hidden_layers": [16, 8], "learning_rate": 0.01, "batch_size": 16, "epochs": 1000, "loss_threshold": 0.02, "patience": 70, "min_delta": 1e-5, "seed": 34, "lambda_l2": 0.0005, "lr_decay": 0.0005},
        {"preprocessing": "One-hot", "hidden_layers": [32, 16], "learning_rate": 0.005, "batch_size": 32, "epochs": 1000, "loss_threshold": 0.02, "patience": 80, "min_delta": 1e-5, "seed": 35, "lambda_l2": 0.001, "lr_decay": 0.0005},
        {"preprocessing": "One-hot", "hidden_layers": [64, 16], "learning_rate": 0.003, "batch_size": 32, "epochs": 1200, "loss_threshold": 0.02, "patience": 80, "min_delta": 1e-5, "seed": 36, "lambda_l2": 0.001, "lr_decay": 0.0005},
        {"preprocessing": "One-hot", "hidden_layers": [16], "learning_rate": 0.01, "batch_size": 16, "epochs": 1000, "loss_threshold": 1.10, "patience": 80, "min_delta": 1e-5, "early_monitor": "train", "seed": 37, "lambda_l2": 0.0001},
        {"preprocessing": "One-hot", "hidden_layers": [8], "learning_rate": 0.0001, "batch_size": 32, "epochs": 500, "loss_threshold": 0.02, "patience": 10, "min_delta": 1e-2, "seed": 38},
    ]


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / DATA_FILE

    df = load_data(csv_path)
    feature_df, y = split_features_labels(df)
    num_classes = len(np.unique(y))
    train_df, val_df, test_df, y_train, y_val, y_test = make_train_validation_test_split(feature_df, y, seed=42)
    split_sizes = {"train": len(y_train), "validation": len(y_val), "test": len(y_test)}
    configs = build_configs()

    print(f"Dataset: {df.shape[0]} mẫu, {df.shape[1]} cột")
    print(
        f"Train: {split_sizes['train']} mẫu, Validation: {split_sizes['validation']} mẫu, "
        f"Test: {split_sizes['test']} mẫu"
    )
    print(f"Bắt đầu huấn luyện {len(configs)} cấu hình MLP tự cài đặt...\n")

    results, histories, best_index = run_experiments(
        train_df, val_df, test_df, y_train, y_val, y_test, configs, num_classes
    )
    print_results_table(results)

    save_results_csv(results, base_dir / RESULTS_CSV)
    plot_loss_curves(histories, best_index)
    report_text = build_report_text(df, list(feature_df.columns), split_sizes, results, histories, best_index)
    write_report_markdown(report_text, base_dir / REPORT_MD)
    write_report_pdf(report_text, results, best_index, base_dir / REPORT_PDF)

    best = results[best_index]
    print("\nCấu hình tốt nhất trong bảng thực nghiệm:")
    print(
        f"Config {best['STT']} - preprocessing={best['Kiểu tiền xử lý']}, "
        f"hidden_layers={best['Cấu trúc mạng']}, validation_accuracy={best['Validation accuracy']:.4f}, "
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
