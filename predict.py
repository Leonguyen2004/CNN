import numpy as np
import os
from PIL import Image as PILImage
from model import CNNModel
import argparse

def predict_single_image(image_path, cnn_model, idx_to_class_map, target_size=(28, 28)):
    try:
        img = PILImage.open(image_path).convert('L')
        img_resized = img.resize(target_size)
        img_array = np.array(img_resized) / 255.0

        probabilities, _ = cnn_model.forward(img_array, training=False) # training=False cho dropout

        predicted_idx = np.argmax(probabilities)
        predicted_char = idx_to_class_map.get(predicted_idx, "Không rõ")
        confidence = probabilities[predicted_idx]

        # print(f"Ảnh: {os.path.basename(image_path)}")
        # print(f"  Xác suất dự đoán: {probabilities}")
        # print(f"  Index dự đoán: {predicted_idx}, Ký tự dự đoán: {predicted_char}, Độ tin cậy: {confidence:.4f}")
        return predicted_char, confidence
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file ảnh tại {image_path}")
        return "Lỗi file", 0.0
    except Exception as e:
        print(f"Lỗi khi xử lý ảnh {image_path} để dự đoán: {e}")
        return "Lỗi xử lý", 0.0

def predict_from_folder(folder_path, model_path):
    cnn_model, idx_to_class_map = CNNModel.load_model(model_path)

    if not cnn_model or not idx_to_class_map:
        print("Không thể tải model. Dừng dự đoán.")
        return

    # Lấy target_size từ model đã load (thông qua input_shape)
    target_size = cnn_model.input_shape
    print(f"Kích thước ảnh mục tiêu (từ model): {target_size}")

    if not os.path.isdir(folder_path):
        print(f"Lỗi: Thư mục '{folder_path}' không tồn tại.")
        return

    print(f"\nBắt đầu dự đoán cho các ảnh trong thư mục: {folder_path}")
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]

    if not image_files:
        print("Không tìm thấy file ảnh nào trong thư mục được chỉ định.")
        return

    for image_name in image_files:
        image_path = os.path.join(folder_path, image_name)
        predicted_char, confidence = predict_single_image(image_path, cnn_model, idx_to_class_map, target_size)
        print(f"Ảnh: {image_name} -> Dự đoán: {predicted_char} (Confidence: {confidence:.2f})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Nhận diện ký tự từ ảnh trong một thư mục sử dụng model CNN đã huấn luyện.")
    parser.add_argument("folder_path", type=str, help="Đường dẫn đến thư mục chứa các ảnh cần nhận diện.")
    
    # Đường dẫn model mặc định
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_path = os.path.join(script_dir, "MyCNNModel", "my_cnn_model_v3.npz")
    # Cho phép người dùng ghi đè đường dẫn model
    parser.add_argument("--model_path", type=str, 
                        default=os.environ.get("MODEL_PATH", default_model_path),
                        help=f"Đường dẫn đến file model đã huấn luyện (.npz). Mặc định: {default_model_path}")

    args = parser.parse_args()
    
    # Đảm bảo đường dẫn model là tuyệt đối hoặc tương đối chính xác
    resolved_model_path = args.model_path
    if not os.path.isabs(resolved_model_path):
        resolved_model_path = os.path.join(script_dir, resolved_model_path)
    
    # In ra đường dẫn model sẽ được sử dụng
    print(f"Sử dụng model tại: {os.path.abspath(resolved_model_path)}")

    predict_from_folder(args.folder_path, resolved_model_path)