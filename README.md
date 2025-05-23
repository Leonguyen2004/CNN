OCRDatasetStandard: https://www.kaggle.com/datasets/preatcher/standard-ocr-dataset

*Cách training trên google colab, cần tải sẵn dataset đặt vào google drive*
!git clone https://github.com/Leonguyen2004/CNN.git
from google.colab import drive
drive.mount('/content/drive')
%cd /content/CNN
!python train.py

*Cách chạy file extracted_plate tại local*
Chỉnh sửa image_path, output_folder bằng path đến ảnh cần phân tách, path đến nơi chứa ảnh được phân tách

*Cách chạy file predict tại local*
python predict.py ./<Tên folder chứa các ảnh ký tự>