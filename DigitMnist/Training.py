
'''Importing keras so we can load and predict the data'''
import tensorflow as tf
from tensorflow import keras
'''Importing Numpy so we can use the argmax function'''
import numpy as np
'''importing matplotlib so we can display the results on the screen.'''
import  matplotlib.pyplot as plt
'''importing os so we can Fetch the data'''
import os

'''Fetching the data'''
model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'digitmodel.h5')
data = keras.datasets.mnist 

'''Seprate the Training data and the testing data'''
(train_images, train_lables), (test_images, test_lables) = data.load_data()
class_names = ['0', '1', '2', '3', '4',
               '5', '6', '7', '8', '9']
train_images = train_images/255.0
test_images = test_images/255.0

'''Training'''
model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
keras.layers.Dense(128, activation="relu"),
keras.layers.Dense(128, activation="relu"),
keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
hist = model.fit(train_images, train_lables, epochs=15)

prediction = model.predict(test_images)
model.save("digitmodel.h5",hist)

'''predicting the results'''
'''model = keras.models.load_model(model_path)
prediction = model.predict(test_images)'''

'''Drawing the prediction, actual result, and the image of the number'''
for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap= plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_lables[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
