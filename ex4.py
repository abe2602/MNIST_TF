#Bruno Bacelar Abe, 9292858
import tensorflow as tf
from tensorflow import keras
import numpy as np

batch_size = 128
num_classes = 10
epochs = 5

#Recupera os dados para treino e test
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

#Ajeita a imagem
if keras.backend.image_data_format() == 'channels_first':
    train_images = train_images.reshape(train_images.shape[0], 1, 28, 28)
    test_images = test_images.reshape(test_images.shape[0], 1, 28, 28)
    input_shape = (1, 28, 28)
else:
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
    input_shape = (28, 28, 1)

#Normaliza os valores
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')
train_images /= 255
test_images /= 255
print('train_images shape:', train_images.shape)
print(train_images.shape[0], 'train samples')
print(test_images.shape[0], 'test samples')

# Converte para matriz
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

#Cria o modelo usando Keras
model = keras.Sequential()
model.add(keras.layers.Conv2D(64, kernel_size=(2, 2),
                 activation='relu',
                 input_shape=(input_shape)))

model.add(keras.layers.Conv2D(128, (2, 2), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(4, 4)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(num_classes, activation='softmax'))

#Compila o modelo
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#Treina o modelo
model.fit(train_images, train_labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(test_images, test_labels))

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Loss: ", test_loss, "Accuracy", test_acc)

#Predição do melhor avaliado
predictions = model.predict(test_images)
maxConfidence = np.argmax(predictions[0])
maxConfidence2 = np.argmax(predictions[1])
maxConfidence3 = np.argmax(predictions[2])
maxConfidence4 = np.argmax(predictions[3])

(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

print("Obtido: ", maxConfidence, "Esperado: ", test_labels[0])
print("Obtido: ", maxConfidence2, "Esperado: ", test_labels[1])
print("Obtido: ", maxConfidence3, "Esperado: ", test_labels[2])
print("Obtido: ", maxConfidence4, "Esperado: ", test_labels[3])
