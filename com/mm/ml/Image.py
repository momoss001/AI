import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np

lena = mpimg.imread('lena.png') # 读取和代码处于同一目录下的 lena.png
# 此时 lena 就已经是一个 np.array 了，可以对它进行任意处理
lena.shape #(512, 512, 3)

plt.imshow(lena) # 显示图片
plt.axis('off') # 不显示坐标轴
plt.show()


from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

import numpy as np

np.random.seed(1671)

from keras.callbacks import Callback


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
         self.losses = [ ]

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

NB_CLASSES = 10
RESHAPED = 784
OPTUMIZER = Adam()
BATCH_SIZE = 128
NB_EPOCH = 20
VERBOSE = 1
VALIDATION_SPLIT = 0.2
#
N_HIDDEN = 128
DROPOUT = 0.3
(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')

#
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)

model = Sequential()
model.add(Dense(N_HIDDEN, input_shape=(RESHAPED,)))
model.add(Activation('relu'))
model.add(Dense(N_HIDDEN))
#model.add(Dropout(DROPOUT))
model.add(Activation('relu'))
model.add(Dense(NB_CLASSES))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=OPTUMIZER, metrics=['accuracy'])
his = LossHistory()

history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT,
                    verbose=VERBOSE
                   , callbacks=[TensorBoard(log_dir="d://tensorBoard")]
                    )

score = model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE)
print("test score", score[0])
print("test accuracy", score[1])

result = model.predict_classes(X_test)




model.save('my_model.h5')

json_string = model.to_json()
yaml_string = model.to_yaml()



