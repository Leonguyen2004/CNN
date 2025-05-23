import numpy as np
import os
from PIL import Image as PILImage
import subprocess

def unzip_dataset(rar_path, unzip_dir):
    if not os.path.exists(unzip_dir):
        os.makedirs(unzip_dir)
        print(f"Thư mục {unzip_dir} đã được tạo.")
    else:
        print(f"Thư mục {unzip_dir} đã tồn tại.")

    # Kiểm tra xem file rar có tồn tại không
    if not os.path.exists(rar_path):
        print(f"Lỗi: File {rar_path} không tồn tại.")
        return False

    print(f"Đang giải nén {rar_path} vào {unzip_dir}...")
    try:
        # Sử dụng -o+ để ghi đè các file đã tồn tại
        process = subprocess.run(['unrar', 'x', '-o+', rar_path, unzip_dir + os.sep], capture_output=True, text=True, check=True)
        print("Giải nén thành công!")
        # print(process.stdout) # In ra output của lệnh unrar nếu cần
        return True
    except FileNotFoundError:
        print("Lỗi: Lệnh 'unrar' không được tìm thấy. Hãy đảm bảo unrar đã được cài đặt và nằm trong PATH.")
        return False
    except subprocess.CalledProcessError as e:
        print("Lỗi khi giải nén:")
        print(e.stderr if e.stderr else e.stdout)
        return False
    except Exception as e:
        print(f"Đã xảy ra lỗi không xác định: {e}")
        return False


def load_custom_data(base_dir, target_size=(28, 28)):
    images = []
    labels = []

    if not os.path.exists(base_dir):
        print(f"Lỗi: Thư mục {base_dir} không tồn tại.")
        return np.array(images), np.array(labels), 0, {}, {}

    class_names = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    if not class_names:
        print(f"Lỗi: Không tìm thấy thư mục lớp nào trong {base_dir}.")
        return np.array(images), np.array(labels), 0, {}, {}

    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    idx_to_class = {idx: class_name for class_name, idx in class_to_idx.items()}

    num_classes = len(class_names)
    print(f"Tìm thấy {num_classes} lớp: {class_names}")
    print(f"Mapping lớp sang index: {class_to_idx}")

    for class_name in class_names:
        class_dir = os.path.join(base_dir, class_name)
        for image_name in os.listdir(class_dir):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_path = os.path.join(class_dir, image_name)
                try:
                    img = PILImage.open(image_path).convert('L')
                    img = img.resize(target_size)
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    labels.append(class_to_idx[class_name])
                except Exception as e:
                    print(f"Lỗi khi tải ảnh {image_path}: {e}")

    if not images:
        print(f"Lỗi: Không tải được ảnh nào từ {base_dir}.")
        return np.array(images), np.array(labels), num_classes, idx_to_class, class_to_idx

    return np.array(images), np.array(labels), num_classes, idx_to_class, class_to_idx