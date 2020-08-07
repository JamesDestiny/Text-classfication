import tensorflow as tf
import time
from dataset.cnew_loader import *

from tensorflow.python.ops import gen_image_ops
#tf.image.non_max_suppression = gen_image_ops.non_max_suppression_v2

time = time.time()
#tf.compat.v1.disable_eager_execution()
class CNN(tf.keras.Model):
    """
    CNN配置参数
    """
    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            filters=128,  # 卷积层神经元（卷积核）数目
            kernel_size=5,  # 感受野大小
            padding='same',  # padding策略（vaild 或 same）
            activation=tf.nn.relu  # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
        """self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        """
        self.flatten = tf.keras.layers.Reshape(target_shape=(200 * 128,))
        #self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):# [batch_size, 1, 400, 1]
        #x = self.conv1(inputs)
        x = self.conv1(inputs) # [batch_size, 1, 400, 128]
        x = self.pool1(x) #[batch_size,0.5,400,128]
        x = self.flatten(x)  # [batch_size, 400]
        x = self.dense1(x)  # [batch_size, 100]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output
"""
vocab_size = 500
embedding_dim = 64
num_epochs = 0.1
batch_size = 50
learning_rate = 0.001
model = CNN()
seq_lenth = 400

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

time = time.time()
train_path = "../dataset/cnews.train.txt"
vocab_path = "../dataset/cnews.vocab.txt"
x_train,y_train = getdata(train_path,vocabdir=vocab_path)
test_path = "../dataset/cnews.test.txt"
x_test,y_test = getdata(test_path,vocabdir=vocab_path)
print(y_train.shape)
y_train = onehot_transfer(y_train)
y_test = onehot_transfer(y_test)
num_batches = int(x_train.shape[0] // batch_size * num_epochs)
print(num_batches)
for batch_index in range(num_batches):
    X= random_getdata(x_train,batch_size)
    y = random_getdata(y_train,batch_size)
    embedding = tf.Variable('embedding', [vocab_size, embedding_dim])
    input_x = tf.compat.v1.placeholder(tf.int32,[X,seq_lenth ],name='input_x')
    embedding_inputs = tf.nn.embedding_lookup(embedding, input_x)
    model.input_x = X
    with tf.GradientTape() as tape:
        y_pred = model(embedding_inputs)
        print(y_pred.shape)
        print(y.shape)
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss)
        print("batch %d: loss %f" % (batch_index, loss.numpy()))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
num_batches = int(x_test.shape[0] // batch_size)
for batch_index in range(num_batches):
    start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
    y_pred = model.predict(x_test)
    sparse_categorical_accuracy.update_state(y_true=y_test, y_pred=y_pred)
print("test accuracy: %f" % sparse_categorical_accuracy.result())
"""

vocab_size = 5000
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 128))
#model.add(tf.keras.layers.Conv1D(filters=6,kernel_size=3,strides=1))
model.add(tf.keras.layers.GlobalAveragePooling1D())
#model.add(tf.keras.layers.MaxPool1D(pool_size=2))
#model.add(tf.keras.layers.Reshape(target_shape=(199 * 6,)))
#model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(100, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='sigmoid'))
model.summary()
model.compile(optimizer='adam',
             loss='categorical_crossentropy',
             metrics=['accuracy'])

train_path = "../dataset/cnews.train.txt"
vocab_path = "../dataset/cnews.vocab.txt"
x_train,y_train = getdata(train_path,vocabdir=vocab_path)
#x_train =lower_data(x_train,100)
#y_train = lower_data(y_train,100)
test_path = "../dataset/cnews.test.txt"
x_test,y_test = getdata(test_path,vocabdir=vocab_path)

#history = model.fit(x_train,y_train,epochs=40, batch_size=500)
#model.save("Cnn_model")
new_model = tf.keras.models.load_model("Cnn_model")
classes = new_model.predict(x_test)
result = new_model.evaluate(x_test, y_test)
print(x_test[0].shape)
print(classes[0].shape)
print(np.argmax(classes[0:10],axis=1))
print(onehot_transfer(y_test[0:10]))

print(result)
print(get_time_dif(time))

"""
print(x_train.shape)
print(y_train.shape)
print(x_train[0:10])
x = tf.keras.layers.Embedding(5000,400)(x_train)
print(x.shape)
x = tf.keras.layers.Conv1D(kernel_size=3, filters=6,strides=1)(x)
print(x.shape)
x= tf.keras.layers.MaxPool1D(pool_size=2)(x)
print(x.shape)
x = tf.keras.layers.Reshape(target_shape=(199 * 6,))(x)
print(x.shape)
x = tf.keras.layers.Dense(1194, activation='relu')(x)
print(x.shape)
x = tf.keras.layers.Dense(10,activation='sigmoid')(x)
print(x.shape)
print(x[0:10])
"""