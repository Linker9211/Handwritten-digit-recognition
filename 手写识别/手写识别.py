from keras.utils import to_categorical
from keras import models, layers, regularizers
from keras.optimizers import RMSprop
from keras.datasets import mnist
import matplotlib.pyplot as plt

# 加载数据集(训练集和测试集)
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
plt.imshow(train_images[21])
plt.show()
#数据集的处理：全连接层输入为一维，reshape二维化一维
train_images = train_images.reshape((60000, 28*28)).astype('float')
test_images = test_images.reshape((10000, 28*28)).astype('float')
#one-hot编码使得数据类别之间的关系解释为对立的关系，消除相对大小带来的错误，提高性能
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
#创建神经网路
network = models.Sequential()
#dense隐藏层创建，层数：128，激活函数：relu，输入：28*28一维向量，kernel_regularizer：l1正则化
network.add(layers.Dense(units=128, activation='relu', input_shape=(28*28, ),
                         kernel_regularizer=regularizers.l1(0.0001)))
#dropout提高泛化能力，防止过拟合
network.add(layers.Dropout(0.01))
network.add(layers.Dense(units=32, activation='relu',
                         kernel_regularizer=regularizers.l1(0.0001)))
network.add(layers.Dropout(0.01))
#输出层使用softmax函数，通常用于最后一层输出，表示为具有概率的向量（[0,1]之间）
network.add(layers.Dense(units=10, activation='softmax'))
# compile方法指定编译做法和评估指标，optimizer是优化器参数，lr表示学习率初始为0.001
#定义两个学习指标：loss和metrics，损失和精度
network.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# 训练网络，用fit函数, epochs表示训练多少个回合， batch_size表示每次训练给多大的数据
# verbose控制训练过程中的输出信息的详细程度。0时，不输出信息；1时，进度条中输出；2时，每个epoch输出一次训练信息
network.fit(train_images, train_labels, epochs=10, batch_size=256, verbose=1)
#将神经网络中定义的两个指标在训练后返回给test_loss,test_accuaracy
test_loss, test_accuracy = network.evaluate(test_images, test_labels)
print("test_loss:", test_loss, "    test_accuracy:", test_accuracy)
