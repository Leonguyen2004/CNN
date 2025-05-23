import numpy as np
import os
from layers import Conv, ReLU, MaxPool, Flatten, Dense, Dropout, Softmax

class CNNModel:
    def __init__(self, input_shape=(28, 28), num_classes=10,
                 conv_filters=8, conv_kernel_size=3,
                 pool_size=2, pool_stride=2,
                 dense1_nodes=128, dropout_rate=0.5):

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conv_filters = conv_filters
        self.conv_kernel_size = conv_kernel_size
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.dense1_nodes = dense1_nodes
        self.dropout_rate = dropout_rate

        # Khởi tạo các layers
        self.conv1 = Conv(num_filters=self.conv_filters, filter_size=self.conv_kernel_size)
        self.relu1 = ReLU()
        self.pool1 = MaxPool(pool_size=self.pool_size, stride=self.pool_stride)
        self.flatten_layer = Flatten()

        # Tính toán kích thước đầu vào cho lớp Dense đầu tiên
        conv_out_h = self.input_shape[0] - self.conv_kernel_size + 1
        conv_out_w = self.input_shape[1] - self.conv_kernel_size + 1
        pool_out_h = (conv_out_h - self.pool_size) // self.pool_stride + 1
        pool_out_w = (conv_out_w - self.pool_size) // self.pool_stride + 1
        self.flattened_size = pool_out_h * pool_out_w * self.conv_filters

        self.dense1 = Dense(input_len=self.flattened_size, output_len=self.dense1_nodes)
        self.relu2 = ReLU()
        self.dropout1 = Dropout(rate=self.dropout_rate)
        self.dense2 = Dense(input_len=self.dense1_nodes, output_len=self.num_classes)
        self.softmax_activation = Softmax()

        print(f"Model initialized with flattened_size={self.flattened_size}, num_classes={self.num_classes}")

    def forward(self, image, training=True):
        img_processed = image - 0.5 # Chuẩn hóa [-0.5, 0.5]

        out = self.conv1.forward(img_processed)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        out = self.flatten_layer.forward(out)
        out = self.dense1.forward(out)
        out = self.relu2.forward(out)
        out = self.dropout1.forward(out, training=training)
        logits = self.dense2.forward(out)
        probs = self.softmax_activation.forward(logits)
        return probs, logits

    def save_model(self, path, idx_to_class_map):
        print(f"Saving model to {path}...")
        # Tạo thư mục nếu chưa tồn tại
        os.makedirs(os.path.dirname(path), exist_ok=True)

        params_to_save = {
            # Tham số kiến trúc
            'img_h': self.input_shape[0],
            'img_w': self.input_shape[1],
            'num_model_classes': self.num_classes,  # Khớp với 'num_model_classes' khi lưu
            'conv_filters_count': self.conv_filters,  # Khớp với 'conv_filters_count' khi lưu
            'conv_kernel_size': self.conv_kernel_size,
            'pool_size': self.pool_size,
            'pool_stride': self.pool_stride,
            'dense1_nodes': self.dense1_nodes,
            'dropout_rate': self.dropout_rate,

            # Trọng số và map
            'conv1_filters': self.conv1.filters,
            'dense1_weights': self.dense1.weights,
            'dense1_biases': self.dense1.biases,
            'dense2_weights': self.dense2.weights,
            'dense2_biases': self.dense2.biases,
            'idx_to_class_map': np.array(list(idx_to_class_map.items()), dtype=object)
        }

        try:
            np.savez(path, **params_to_save)
            print(f"Model đã được lưu thành công vào: {path}")
        except Exception as e:
            print(f"Lỗi khi lưu model: {e}")

    @staticmethod
    def load_model(path):
        print(f"Loading model from {path}...")
        if not os.path.exists(path):
            print(f"Error: Model file not found at {path}")
            return None, None

        try:
            data = np.load(path, allow_pickle=True)

            model = CNNModel(input_shape=(int(data['img_h']), int(data['img_w'])),
                             num_classes=int(data['num_model_classes']),
                             conv_filters=int(data['conv_filters_count']),
                             conv_kernel_size=int(data['conv_kernel_size']),
                             pool_size=int(data['pool_size']),
                             pool_stride=int(data['pool_stride']),
                             dense1_nodes=int(data['dense1_nodes']),
                             dropout_rate=float(data['dropout_rate']))

            # Tên khóa cho trọng số có vẻ đã đúng (ví dụ: conv1_filters_val)
            # Nhưng để nhất quán với logic lưu ở trên, có thể bạn đã lưu là conv1_filters
            # Kiểm tra lại file model của bạn lưu trọng số với tên gì.
            # Giả sử file lưu của bạn dùng tên `conv1_filters`, `dense1_weights` v.v. (không có _val)
            # Nếu file lưu là `conv1_filters` (như trong đoạn code bạn cung cấp)
            model.conv1.filters = data['conv1_filters']
            model.dense1.weights = data['dense1_weights']
            model.dense1.biases = data['dense1_biases']
            model.dense2.weights = data['dense2_weights']
            model.dense2.biases = data['dense2_biases']

            idx_to_class_map = {item[0]: item[1] for item in
                                data['idx_to_class_map']}  # Giả sử khóa là 'idx_to_class_map'
            print("Model loaded successfully.")
            return model, idx_to_class_map
        except KeyError as e:
            print(f"Error loading model: Missing key {e} in the model file.")
            print("Please ensure the model file was saved with all necessary architecture parameters and weights.")
            import traceback
            traceback.print_exc()
            return None, None
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return None, None