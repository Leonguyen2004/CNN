import cv2
import numpy as np
import os
from pathlib import Path
from skimage import measure, filters
from skimage.measure import regionprops
import string

def extract_characters_improved(image_path, output_folder='extracted_chars'):
    # Tạo thư mục đầu ra
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return []

    # Tiền xử lý ảnh
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Xử lý morphology
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # Phân tích connected components
    labels = measure.label(thresh, connectivity=2)
    regions = regionprops(labels)

    # Lọc vùng ký tự
    char_regions = []
    for region in regions:
        y1, x1, y2, x2 = region.bbox
        w, h = x2 - x1, y2 - y1
        aspect_ratio = w / float(h)
        area = region.area

        if area > 200 and 0.1 < aspect_ratio < 1.5 and h > height * 0.2:
            char_regions.append((x1, y1, w, h))

    # Sắp xếp từ trái sang phải
    char_regions.sort(key=lambda x: x[0])

    # Lưu các ký tự
    for i, (x, y, w, h) in enumerate(char_regions):
        padding = 5
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(width, x + w + padding)
        y_end = min(height, y + h + padding)

        char_img = img[y_start:y_end, x_start:x_end]
        output_path = os.path.join(output_folder, f'char_{i}.png')
        cv2.imwrite(output_path, char_img)

    return char_regions

IMG_SIZE = (28, 28)
class_names = list(string.digits) + list(string.ascii_uppercase)

def predict_character(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # Thêm batch và channel dimensions
    pred = model.predict(img)
    return class_names[np.argmax(pred)]

if __name__ == "__main__":
    # # Load model
    # model = load_model('best_model.keras')  # Đảm bảo file model tồn tại

    # Phân tách ký tự
    image_path = 'plate_image/97WW755.png'
    output_folder = 'extracted_chars'
    chars = extract_characters_improved(image_path, output_folder)

