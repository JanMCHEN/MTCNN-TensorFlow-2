import tensorflow as tf


def softmax_cross_entropy(y, y_):
    y_ = tf.squeeze(y_)
    y = tf.squeeze(y)

    # 只计算pos和neg的损失
    pos_neg = tf.where(y >= 0)

    y_ = tf.gather(y_, pos_neg[:, 0])
    y = tf.gather(y, pos_neg[:, 0])
    y = tf.cast(y, tf.int32)

    # return tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_)
    return tf.losses.sparse_categorical_crossentropy(y, y_, axis=-1)


def mse(y, y_):
    pp = tf.where(tf.not_equal(y[:, 0], 0))

    y_ = tf.gather(tf.squeeze(y_), pp[:, 0])
    y = tf.gather(y[:, 1:], pp[:, 0])

    error = tf.square(y_ - y)

    return tf.reduce_mean(error)


def accuracy(y, y_):
    y_ = tf.argmax(tf.squeeze(y_), axis=1)
    y = tf.squeeze(tf.abs(y))
    right = tf.cast(tf.equal(y_, tf.cast(y, tf.int64)), dtype=tf.int32)

    return tf.reduce_sum(right) / tf.shape(right)[0]


class PNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layers_ = tf.keras.Sequential([
            tf.keras.layers.Conv2D(10, (3, 3), input_shape=(None, None, 3), activation='relu',
                                   kernel_initializer=tf.initializers.glorot_normal,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.MaxPool2D(),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                                   kernel_initializer=tf.initializers.glorot_normal),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                   kernel_initializer=tf.initializers.glorot_normal),
            # tf.keras.layers.BatchNormalization(),

        ])

        self.face_class = tf.keras.layers.Conv2D(2, (1, 1), activation='softmax')
        self.box_reg = tf.keras.layers.Conv2D(4, (1, 1))

        self.compile(optimizer='adam', loss=[softmax_cross_entropy, mse],
                     metrics=[[accuracy], []], loss_weights=[1, 0.5])

    def call(self, inputs):
        x = self.layers_(inputs)
        cls = self.face_class(x)
        reg = self.box_reg(x)

        return cls, reg

    def detect(self, inputs):
        x = self.layers_(inputs)
        return self.face_class(x)


class RNet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layers_ = tf.keras.Sequential([
            tf.keras.layers.Conv2D(28, (3, 3), input_shape=(24, 24, 3), activation='relu',
                                   kernel_initializer=tf.initializers.glorot_normal,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2)),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(48, (3, 3), activation='relu',
                                   kernel_initializer=tf.initializers.glorot_normal),
            tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(64, (2, 2), activation='relu',
                                   kernel_initializer=tf.initializers.glorot_normal),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128),

        ])
        self.face_class = tf.keras.layers.Dense(2, activation='softmax')
        self.box_reg = tf.keras.layers.Dense(4)

        self.compile(optimizer='adam', loss=[softmax_cross_entropy, mse],
                     metrics=[[accuracy], []], loss_weights=[1, 0.8])

    def call(self, inputs):
        x = self.layers_(inputs)
        cls = self.face_class(x)
        reg = self.box_reg(x)

        return cls, reg


class ONet(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.layers_ = self.layers_ = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), input_shape=(48, 48, 3), activation='relu',
                                   kernel_initializer=tf.initializers.glorot_normal,
                                   kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2)),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                   kernel_initializer=tf.initializers.glorot_normal),
            tf.keras.layers.MaxPool2D((3, 3), strides=(2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                                   kernel_initializer=tf.initializers.glorot_normal),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, (2, 2), activation='relu',
                                   kernel_initializer=tf.initializers.glorot_normal),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256),

        ])
        self.face_class = tf.keras.layers.Dense(2, activation='softmax')
        self.box_reg = tf.keras.layers.Dense(4)

        self.compile(optimizer='adam', loss=[softmax_cross_entropy, mse],
                     metrics=[[accuracy], []], loss_weights=[1, 0.8])

    def call(self, inputs):
        x = self.layers_(inputs)
        cls = self.face_class(x)
        reg = self.box_reg(x)

        return cls, reg


if __name__ == '__main__':
    from data.tf_data import dataset_gen, LABEL_SAVE_FILE, CROP_SAVE_12

    model = PNet()
    # dataset = dataset_gen(LABEL_SAVE_FILE, CROP_SAVE_12)
    #
    # dataset = dataset.batch(4)
    #
    # model.fit(dataset, epochs=2, steps_per_epoch=10)


