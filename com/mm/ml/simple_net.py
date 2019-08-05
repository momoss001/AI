from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt

# 数据分类类别
NB_CLASSES = 10
RESHAPED = 784
# 优化器
OPTUMIZER = Adam()
# 每批训练输入数据大小
BATCH_SIZE = 128
# 训练轮数
NB_EPOCH = 100
# 验证集在训练集中所占百分比
VALIDATION_SPLIT = 0.2
#
N_HIDDEN = 128
DROPOUT = 0.3

#读取数据集
# X_train为训练样本， y_train 为训练样本结果集
# X_test为训练样本， y_test 为训练样本结果集
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# 将数据集转换为可供数据
x_train = X_train.reshape(60000, RESHAPED)
x_test = X_test.reshape(10000, RESHAPED)

# 类型转换为float32
x_train = X_train.astype('float32')
x_test = X_test.astype('float32')

# 像素的亮度值
x_train /= 255
x_test /= 255

#打印下训练数据
print(x_train.shape[0], 'train samples')


#将训练结果集及数据集进行oneHot 编码
y_train = np_utils.to_categorical(Y_train, NB_CLASSES)
y_test = np_utils.to_categorical(Y_test, NB_CLASSES)

# 构建神经网络模型
# 输入为 RESHAPED：784 隐藏层为N_HIDDEN：128  分类类别为10的神经网络模型
# 创建一个序贯模型
model = Sequential()

# 输入层  网络结构为 784
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
# 激活函数
model.add(Activation('relu'))

# 添加隐藏层
model.add(Dense(N_HIDDEN))
model.add(Dropout(DROPOUT))
# 激活函数
model.add(Activation('relu'))

model.add(Dense(N_HIDDEN))
#model.add(Dropout(DROPOUT))
# 激活函数
#model.add(Activation('relu'))

# 输出层
model.add(Dense(NB_CLASSES))
# 归一化函数， 将结果映射到数据分类上
model.add(Activation('softmax'))

# loss 损失函数 optimizer 优化器 metrics 衡量指标
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=OPTUMIZER, metrics=['accuracy'])

# 训练数据
# validation_split：百分比; 训练集中的部分数据作为验证集
# epochs 训练轮数 batch_size 每次训练的批次
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=20, validation_split=VALIDATION_SPLIT)

# 评估模型在测试集上的表现
score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print("test score", score[0])
print("test accuracy", score[1])

#预测测试集的结果
result = model.predict_classes(x_test)
plt.imshow(X_test[9])
plt.show()
print(result[9])




#保存模型，可以下次使用
model.save('my_model.h5')
json_string = model.to_json()
yaml_string = model.to_yaml()



