import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

load_save = True

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

if not load_save:
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # input 1
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu)) # input 2
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax)) # output layer

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=3)

    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss, val_acc)

    model.save('num_reader.model')
else:
    loaded_model = tf.keras.models.load_model('num_reader.model')
    os.system("cls")


predictions = loaded_model.predict(x_test)

index = 0
while index < 9:
    print(np.argmax(predictions[index]))

    plt.imshow(x_test[index])
    plt.show()
    index+=1
