from dataset.cnew_loader import *
time = time.time()

vocab_size = 5000
#使用变长LSTM序列模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Embedding(vocab_size, 128))
#model.add(tf.keras.layers.Conv1D(filters=6,kernel_size=3,strides=1))
#model.add(tf.keras.layers.GlobalAveragePooling1D())
#model.add(tf.keras.layers.MaxPool1D(pool_size=2))
#model.add(tf.keras.layers.Reshape(target_shape=(199 * 6,)))
#model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.SimpleRNN(128))
model.add(tf.keras.layers.Dense(64, activation='relu'))
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

#history = model.fit(x_train,y_train,epochs=6)
#model.save("Rnn_model")
new_model = tf.keras.models.load_model("Rnn_model")
classes = new_model.predict(x_test)
result = new_model.evaluate(x_test, y_test)
print(classes.shape)
print(np.argmax(classes[0:10],axis=1))
print(onehot_transfer(y_test[0:10]))

print(result)
print(get_time_dif(time))
