import os.path
import tensorflow as tf
import utils.constant
from tensorflow.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

train_rices_images_dir = os.path.join(utils.constant.get_rices_images())
train_others_images_dir = os.path.join(utils.constant.get_others_images())

train_rices_images_names = os.listdir(str(train_rices_images_dir))
train_others_images_names = os.listdir(str(train_others_images_dir))

print('total training rices:', len(train_rices_images_names))
print('total training others images:', len(train_others_images_names))

# Preprocessing the Training Data

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1 / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   rotation_range=30,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   fill_mode='nearest',
                                   horizontal_flip=True)
#
# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    utils.constant.all_rices_others_dir(),  # This is the source directory for training images
    target_size=(300, 300),  # All images will be resized to 150x150
    batch_size=100,
    color_mode='rgb',
    class_mode='binary'
)

# Fill and Preprocessing the Test Data

test_datagen = ImageDataGenerator(rescale=1. / 255,)

test_generator = test_datagen.flow_from_directory(
    utils.constant.all_rices_others_test_dir(),
    target_size=(300, 300),
    batch_size=100,
    color_mode='rgb',
    class_mode='binary'
)

#
#
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.5),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 4 output neuron.
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# # tensorboard_model = TensorBoard(log_dir='logs/')
# #
# #
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#
#
history = model.fit(
    x=train_generator,
    validation_data=test_generator,
    epochs=50,
    batch_size=100,
    validation_split=0.20,
    verbose=1)
loss = history.history['loss']
acc = history.history['accuracy']

print(f'History: ${history}')
print(f'Loss: ${loss}')
print(f'Accuracy: ${acc}')
print(f'Class: ${train_generator.class_indices}')
#
model.save(utils.constant.all_states_test_dir() + "validation_rice.h5")
