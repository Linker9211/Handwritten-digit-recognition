from keras.utils import to_categorical
from keras import models, layers
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.backend import clear_session
from keras.datasets import mnist
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
# 设置随机种子，保证实验结果可重现
np.random.seed(0)
# 加载数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 将像素值缩放到 [0, 1] 范围
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# 增加一维通道数，变成四维张量
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# 将标签转换为 one-hot 编码
num_classes = 10
train_labels = to_categorical(train_labels, num_classes)
test_labels = to_categorical(test_labels, num_classes)

# 定义 LeNet 网络
def LeNet():
    network = models.Sequential()
    network.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    network.add(layers.BatchNormalization())
    network.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    network.add(layers.BatchNormalization())
    network.add(layers.MaxPooling2D((2, 2)))
    network.add(layers.Dropout(0.25))
    network.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    network.add(layers.BatchNormalization())
    network.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    network.add(layers.BatchNormalization())
    network.add(layers.MaxPooling2D((2, 2)))
    network.add(layers.Dropout(0.25))
    network.add(layers.Flatten())
    network.add(layers.Dense(units=512, activation='relu'))
    network.add(layers.Dropout(0.5))
    network.add(layers.Dense(units=num_classes, activation='softmax'))
    return network

# 定义学习率衰减函数
def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 20:
        lr *= 0.5 ** (epoch // 20)
    return lr

# 设置训练参数
batch_size = 256
epochs = 10

# 使用数据增强生成器
train_datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                                   zoom_range=0.1, horizontal_flip=True)
train_datagen.fit(train_images)

# 训练模型
clear_session()#多次训练模型需加上这个保证空间的释放
model = LeNet()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr_schedule(0)), metrics=['accuracy'])
model.fit_generator(train_datagen.flow(train_images, train_labels, batch_size=batch_size),
                    steps_per_epoch=len(train_images) / batch_size, epochs=epochs,
                    validation_data=(test_images, test_labels),
                    callbacks=[LearningRateScheduler(lr_schedule)])

# 测试模型
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("test_loss:", test_loss, "test_accuracy:", test_accuracy)

# 预测样本并展示部分结果
predict_labels = np.argmax(model.predict(test_images), axis=-1)
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(test_images[i, :, :, 0], cmap='gray')
    ax.set_title("Predict:{}".format(predict_labels[i]))
plt.show()
