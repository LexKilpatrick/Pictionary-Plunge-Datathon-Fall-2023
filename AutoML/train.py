import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from matplotlib import pyplot as plt
import numpy as np
from DataStream import DataStream

stream=DataStream(10)
images=[]
labels=[]
for image in (stream.get_random_data()):
    images.append(image[0])
    labels.append(image[1])

images=np.array(images)
labels=np.array(labels)

# Define the ratio for the split (80% training, 20% validation)
train_ratio = 0.8
validation_ratio = 0.2

# Calculate the number of images for training and validation
num_images = len(images)
num_train_images = int(train_ratio * num_images)
num_validation_images = num_images - num_train_images

# Split the data into training and validation sets
x_train = images[:num_train_images]
y_train = labels[:num_train_images]
x_test = images[num_train_images:]
y_test = labels[num_train_images:]


# Define the model
model = models.Sequential()

# Add convolutional layers
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten the output and add dense layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(15, activation='softmax'))  # 10 output classes

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()



# 2. Train the model
model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_test, y_test))

model.save('model.h5')
# 3. Predict with the model
predictions = model.predict(x_test)

for i in range(len(predictions)):
    print("Model predicts: ",list(stream.fileNames.keys())[list(stream.fileNames.values()).index(tf.argmax(predictions[i]).numpy())], "Actual: ", list(stream.fileNames.keys())[list(stream.fileNames.values()).index(y_test[i])])
    plt.imshow(np.array(x_test[i], dtype='float'), cmap='gray')
    plt.show()
