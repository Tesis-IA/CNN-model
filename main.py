# Get training data
import os.path
import tensorflow as tf
import utils.constant
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

train_healthy_rice_dir = os.path.join(utils.constant.healthy_dir())
train_leaf_smut_rice_dir = os.path.join(utils.constant.leaf_smut_dir())
train_brown_spot_rice_dir = os.path.join(utils.constant.brown_spot_dir())
train_bacterial_leaf_blight_rice_dir = os.path.join(utils.constant.bacterial_leaf_blight_dir())

train_healthy_rice_names = os.listdir(train_healthy_rice_dir)
train_leaf_smut_rice_names = os.listdir(train_leaf_smut_rice_dir)
train_brown_spot_rice_names = os.listdir(train_brown_spot_rice_dir)
train_bacterial_leaf_blight_rice_names = os.listdir(train_bacterial_leaf_blight_rice_dir)

print('total training healthy rices:', len(train_healthy_rice_names))
print('total training leaf smut images:', len(train_leaf_smut_rice_names))
print('total training brown spot images:', len(train_brown_spot_rice_names))
print('total training bacterial leaf blight images:', len(train_bacterial_leaf_blight_rice_names))

# Parameters for our graph; we'll output images in a 4x4 configuration
n_rows = 4
n_cols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(n_cols * 4, n_rows * 4)

pic_index += 4
next_healthy_pix = [os.path.join(train_healthy_rice_dir, f_name)
                    for f_name in train_healthy_rice_names[pic_index - 4:pic_index]]
next_leaf_smut_pix = [os.path.join(train_leaf_smut_rice_dir, f_name)
                      for f_name in train_leaf_smut_rice_names[pic_index - 4:pic_index]]
next_brown_spot_pix = [os.path.join(train_brown_spot_rice_dir, f_name)
                       for f_name in train_brown_spot_rice_names[pic_index - 4:pic_index]]
next_bacterial_leaf_blight_pix = [os.path.join(train_bacterial_leaf_blight_rice_dir, f_name)
                                  for f_name in train_bacterial_leaf_blight_rice_names[pic_index - 4:pic_index]]

for i, img_path in enumerate(next_healthy_pix + next_leaf_smut_pix + next_brown_spot_pix +
                             next_bacterial_leaf_blight_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(n_rows, n_cols, i + 1)
    sp.axis('Off')  # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other (
    # 'humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['acc'])

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1. / 255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    utils.constant.all_states_dir(),  # This is the source directory for training images
    target_size=(300, 300),  # All images will be resized to 150x150
    batch_size=16,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='binary')

history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    verbose=1)
