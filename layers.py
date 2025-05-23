import numpy as np

class Conv:
    def __init__(self, num_filters, filter_size=3):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_filters, filter_size, filter_size) / (filter_size * filter_size)
        self.padding = 0

    def iterate_regions(self, image):
        h, w = image.shape
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                im_region = image[i:(i + self.filter_size), j:(j + self.filter_size)]
                yield im_region, i, j

    def forward(self, input_data):
        self.last_input = input_data
        h, w = input_data.shape
        output_h = h - self.filter_size + 1
        output_w = w - self.filter_size + 1
        output = np.zeros((output_h, output_w, self.num_filters))

        for im_region, i, j in self.iterate_regions(input_data):
            output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))
        return output

    def backprop(self, d_l_d_out, learn_rate):
        d_l_d_filters = np.zeros(self.filters.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_l_d_filters[f] += d_l_d_out[i, j, f] * im_region
        self.filters -= learn_rate * d_l_d_filters
        return None

class ReLU:
    def forward(self, input_data):
        self.last_input = input_data
        return np.maximum(0, input_data)

    def backprop(self, d_l_d_out):
        return d_l_d_out * (self.last_input > 0)

class MaxPool:
    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride

    def iterate_regions(self, image):
        h, w, num_filters = image.shape
        new_h = (h - self.pool_size) // self.stride + 1
        new_w = (w - self.pool_size) // self.stride + 1
        for i in range(new_h):
            for j in range(new_w):
                im_region = image[(i * self.stride):(i * self.stride + self.pool_size),
                                  (j * self.stride):(j * self.stride + self.pool_size)]
                yield im_region, i, j

    def forward(self, input_data):
        self.last_input = input_data
        h, w, num_filters = input_data.shape
        output_h = (h - self.pool_size) // self.stride + 1
        output_w = (w - self.pool_size) // self.stride + 1
        output = np.zeros((output_h, output_w, num_filters))

        for im_region, i, j in self.iterate_regions(input_data):
            output[i, j] = np.amax(im_region, axis=(0, 1))
        return output

    def backprop(self, d_l_d_out):
        d_l_d_input = np.zeros(self.last_input.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            h_r, w_r, num_filters_r = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))
            for r_i in range(h_r):
                for r_j in range(w_r):
                    for f_k in range(num_filters_r):
                        if im_region[r_i, r_j, f_k] == amax[f_k]:
                            input_i = i * self.stride + r_i
                            input_j = j * self.stride + r_j
                            d_l_d_input[input_i, input_j, f_k] += d_l_d_out[i, j, f_k]
        return d_l_d_input

class Flatten:
    def forward(self, input_data):
        self.last_input_shape = input_data.shape
        return input_data.flatten()

    def backprop(self, d_l_d_out):
        return d_l_d_out.reshape(self.last_input_shape)

class Dense:
    def __init__(self, input_len, output_len):
        self.weights = np.random.randn(input_len, output_len) / np.sqrt(input_len)
        self.biases = np.zeros(output_len)

    def forward(self, input_data):
        self.last_input = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backprop(self, d_l_d_out, learn_rate):
        d_l_d_weights = np.outer(self.last_input, d_l_d_out)
        d_l_d_biases = d_l_d_out
        d_l_d_input = np.dot(d_l_d_out, self.weights.T)
        self.weights -= learn_rate * d_l_d_weights
        self.biases -= learn_rate * d_l_d_biases
        return d_l_d_input

class Dropout:
    def __init__(self, rate):
        self.rate = rate
        self.mask = None

    def forward(self, input_data, training=True):
        if training:
            self.mask = (np.random.rand(*input_data.shape) > self.rate) / (1.0 - self.rate)
            return input_data * self.mask
        else:
            return input_data

    def backprop(self, d_l_d_out):
        return d_l_d_out * self.mask

class Softmax:
    def forward(self, input_data):
        self.last_input_logits = input_data
        exp_shifted = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))
        self.last_output_probs = exp_shifted / np.sum(exp_shifted, axis=-1, keepdims=True)
        return self.last_output_probs

    def backprop(self, d_l_d_out_probs):
        # dL/dz_k = p_k * (dL/dp_k - sum_i(dL/dp_i * p_i))
        p = self.last_output_probs
        dL_dlogits = p * (d_l_d_out_probs - np.sum(d_l_d_out_probs * p, axis=-1, keepdims=True))
        return dL_dlogits