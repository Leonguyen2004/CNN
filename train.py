import numpy as np
import os
from input import load_custom_data, unzip_dataset
from model import CNNModel

# --- Cấu hình đường dẫn ---
GOOGLE_DRIVE_PATH = "/content/drive/MyDrive/Colab Notebooks/" # Ví dụ cho Colab
RAR_FILE_NAME = "OCRDatasetStandard.rar" # Tên file .rar của bạn
MODEL_SAVE_DIR = os.path.join(GOOGLE_DRIVE_PATH, "MyCNNModel") # Thư mục lưu model
MODEL_NAME = "my_cnn_model_v4.npz"

rar_path = "/content/drive/MyDrive/Colab Notebooks/OCRDatasetStandard.rar"
unzip_dir = "/content/dataset" # Giải nén vào /content/dataset trên Colab

train_data_dir = "/content/dataset/data/training_data"
test_data_dir = "/content/dataset/data/testing_data"

model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)


def train_step(cnn_model, image, label, learn_rate=0.005):
    probabilities, logits = cnn_model.forward(image, training=True)

    epsilon = 1e-9
    loss = -np.log(probabilities[label] + epsilon)
    acc = 1 if np.argmax(probabilities) == label else 0

    if np.isinf(loss) or np.isnan(loss):
        # print(f"Loss is {loss} for label {label}, probs: {probabilities}. Skipping backprop.")
        return loss, acc

    grad_logits = probabilities.copy()
    grad_logits[label] -= 1

    grad_dense2_input = cnn_model.dense2.backprop(grad_logits, learn_rate)
    grad_dropout_input = cnn_model.dropout1.backprop(grad_dense2_input)
    grad_relu2_input = cnn_model.relu2.backprop(grad_dropout_input)
    grad_dense1_input = cnn_model.dense1.backprop(grad_relu2_input, learn_rate)
    grad_flatten_input = cnn_model.flatten_layer.backprop(grad_dense1_input)
    grad_pool1_input = cnn_model.pool1.backprop(grad_flatten_input)
    grad_relu1_input = cnn_model.relu1.backprop(grad_pool1_input)
    cnn_model.conv1.backprop(grad_relu1_input, learn_rate)
    return loss, acc

def main():
    if not os.path.exists(train_data_dir) or not os.path.exists(test_data_dir):
        print(f"Thư mục dataset '{train_data_dir}' hoặc '{test_data_dir}' không tìm thấy.")
        print(f"Thử giải nén từ '{rar_path}' vào '{unzip_dir}'.")
        if unzip_dataset(rar_path, unzip_dir):
             print("Giải nén dataset thành công.")
        else:
            print("Không thể giải nén dataset. Vui lòng kiểm tra đường dẫn file .rar và cài đặt unrar.")
            print(f"Đường dẫn file RAR được kiểm tra: {os.path.abspath(rar_path)}")
            print(f"Kỳ vọng thư mục huấn luyện tại: {os.path.abspath(train_data_dir)}")
            return # Thoát nếu không giải nén được
    else:
        print("Tìm thấy thư mục dataset đã giải nén.")


    # 2. Tải dữ liệu
    print("Đang tải dữ liệu huấn luyện...")
    train_images, train_labels, num_classes, idx_to_class, class_to_idx = load_custom_data(train_data_dir)
    # print("Đang tải dữ liệu kiểm tra...") # Dữ liệu test không dùng trong quá trình train này
    # test_images, test_labels, _, _, _ = load_custom_data(test_data_dir)

    if num_classes == 0 or train_images.size == 0:
        print("Không có dữ liệu để huấn luyện. Vui lòng kiểm tra đường dẫn và cấu trúc dataset.")
        return

    IMG_H, IMG_W = train_images[0].shape
    print(f"Kích thước ảnh đầu vào: {IMG_H}x{IMG_W}")
    print(f"Số lượng lớp: {num_classes}")

    # 3. Khởi tạo model
    # Các tham số này có thể được điều chỉnh
    cnn_model = CNNModel(input_shape=(IMG_H, IMG_W),
                         num_classes=num_classes,
                         conv_filters=8,
                         conv_kernel_size=3,
                         pool_size=2,
                         pool_stride=2,
                         dense1_nodes=128,
                         dropout_rate=0.5)

    # 4. Huấn luyện model
    epochs = 5
    learn_rate = 0.001
    batch_print_size = 100

    print('Bắt đầu huấn luyện model...')
    for epoch in range(epochs):
        print(f'---- EPOCH {epoch + 1}/{epochs} ----')
        permutation = np.random.permutation(len(train_images))
        current_train_images = train_images[permutation]
        current_train_labels = train_labels[permutation]

        loss_val = 0
        num_correct = 0

        for i, (im, label) in enumerate(zip(current_train_images, current_train_labels)):
            l, acc = train_step(cnn_model, im, label, learn_rate)

            if not (np.isinf(l) or np.isnan(l)):
                loss_val += l
            num_correct += acc

            if (i + 1) % batch_print_size == 0:
                avg_loss = loss_val / batch_print_size
                accuracy_percent = (num_correct / batch_print_size) * 100
                print(f'[Step {i + 1}/{len(train_images)}] Avg Loss: {avg_loss:.3f} | Accuracy: {accuracy_percent:.2f}%')
                loss_val = 0
                num_correct = 0
        
        # In thông số cuối epoch nếu số lượng sample không chia hết cho batch_print_size
        remaining_steps = len(train_images) % batch_print_size
        if remaining_steps > 0 and len(train_images) > batch_print_size :
            avg_loss = loss_val / remaining_steps if loss_val !=0 else 0
            accuracy_percent = (num_correct / remaining_steps) * 100 if num_correct !=0 else 0
            print(f'[Step {len(train_images)}/{len(train_images)}] Avg Loss (last {remaining_steps} steps): {avg_loss:.3f} | Accuracy: {accuracy_percent:.2f}%')


    print("Huấn luyện hoàn tất.")

    # 5. Lưu model
    if idx_to_class: # Đảm bảo idx_to_class không rỗng
        cnn_model.save_model(model_save_path, idx_to_class)
    else:
        print("Không thể lưu model: idx_to_class rỗng.")

if __name__ == '__main__':
    # Tạo thư mục Google Drive giả nếu chạy cục bộ và chưa có
    # if not os.path.exists(GOOGLE_DRIVE_PATH) and "./" in GOOGLE_DRIVE_PATH:
    #     os.makedirs(GOOGLE_DRIVE_PATH, exist_ok=True)
    # if not os.path.exists(os.path.join(GOOGLE_DRIVE_PATH, "Colab Notebooks")) and "./" in GOOGLE_DRIVE_PATH:
    #    os.makedirs(os.path.join(GOOGLE_DRIVE_PATH, "Colab Notebooks"), exist_ok=True)

    main()