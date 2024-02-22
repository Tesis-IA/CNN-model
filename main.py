import os.path
import tensorflow as tf
import utils.constant
from tensorflow.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

train_all_data_dir = os.path.join(utils.constant.all_states_training_dir())
train_healthy_rice_dir = os.path.join(utils.constant.healthy_dir())
train_leaf_smut_rice_dir = os.path.join(utils.constant.leaf_smut_dir())
train_brown_spot_rice_dir = os.path.join(utils.constant.brown_spot_dir())
train_bacterial_leaf_blight_rice_dir = os.path.join(utils.constant.bacterial_leaf_blight_dir())
train_leaf_scald_dir = os.path.join(utils.constant.leaf_scald_dir())
train_neck_blast_paddy_dir = os.path.join(utils.constant.neck_blast_paddy_dir())
train_rice_sheath_blight_dir = os.path.join(utils.constant.rice_sheath_blight_dir())
train_tungro_dir = os.path.join(utils.constant.tungro_dir())
train_leaf_blast_dir = os.path.join(utils.constant.leaf_blast_dir())
train_hispa_dir = os.path.join(utils.constant.hispa_dir())
rice_stem_rot_dir = os.path.join(utils.constant.rice_stem_rot_dir())

train_all_data_names = os.listdir(train_all_data_dir)
train_healthy_rice_names = os.listdir(train_healthy_rice_dir)
train_leaf_smut_rice_names = os.listdir(train_leaf_smut_rice_dir)
train_brown_spot_rice_names = os.listdir(train_brown_spot_rice_dir)
train_bacterial_leaf_blight_rice_names = os.listdir(train_bacterial_leaf_blight_rice_dir)
train_leaf_scald_names = os.listdir(train_leaf_scald_dir)
train_neck_blast_paddy_names = os.listdir(train_neck_blast_paddy_dir)
train_rice_sheath_blight_names = os.listdir(train_rice_sheath_blight_dir)
train_tungro_names = os.listdir(train_tungro_dir)
train_leaf_blast_names = os.listdir(train_leaf_blast_dir)
train_hispa_names = os.listdir(train_hispa_dir)
rice_stem_rot_names = os.listdir(rice_stem_rot_dir)


print('total training healthy rices:', len(train_healthy_rice_names))
print('total training leaf smut images:', len(train_leaf_smut_rice_names))
print('total training brown spot images:', len(train_brown_spot_rice_names))
print('total training bacterial leaf blight images:', len(train_bacterial_leaf_blight_rice_names))
print('total training leaf scald rices:', len(train_leaf_scald_names))
print('total training neck blast images:', len(train_neck_blast_paddy_names))
print('total training rice sheath blight images:', len(train_rice_sheath_blight_names))
print('total training tungro images:', len(train_tungro_names))
print('total training leaf blast rices:', len(train_leaf_blast_names))
print('total training hispa images:', len(train_hispa_names))
print('total training rice stem rot images:', len(rice_stem_rot_names))

# Parameters for our graph; we'll output images in a 4x4 configuration
n_rows = 4
n_cols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig, and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(n_cols * 4, n_rows * 4)

pic_index += 4
next_healthy_pix = [os.path.join(str(train_healthy_rice_dir), f_name)
                    for f_name in train_healthy_rice_names[pic_index - 4:pic_index]]
next_leaf_smut_pix = [os.path.join(str(train_leaf_smut_rice_dir), f_name)
                      for f_name in train_leaf_smut_rice_names[pic_index - 4:pic_index]]
next_brown_spot_pix = [os.path.join(str(train_brown_spot_rice_dir), f_name)
                       for f_name in train_brown_spot_rice_names[pic_index - 4:pic_index]]
next_bacterial_leaf_blight_pix = [os.path.join(str(train_bacterial_leaf_blight_rice_dir), f_name)
                                  for f_name in train_bacterial_leaf_blight_rice_names[pic_index - 4:pic_index]]

for i, img_path in enumerate(next_healthy_pix + next_leaf_smut_pix + next_brown_spot_pix +
                             next_bacterial_leaf_blight_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(n_rows, n_cols, i + 1)
    sp.axis('Off')  # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)

    plt.imshow(img)

plt.show()

# Preprocessing the Training Data

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1 / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
#
# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    utils.constant.all_states_training_dir(),  # This is the source directory for training images
    target_size=(300, 300),  # All images will be resized to 150x150
    batch_size=32,
    color_mode='rgb',
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode='categorical'
)


# Fill and Preprocessing the Test Data

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    utils.constant.all_states_test_dir(),
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical'
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
    # The third convolution
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The fifth convolution
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Dropout(0.5),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 4 output neuron.
    tf.keras.layers.Dense(11, activation='softmax')
])

model.summary()

tensorboard_model = TensorBoard(log_dir='logs/')


#
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
#
#
history = model.fit(
    x=train_generator,
    validation_data=test_generator,
    epochs=100,
    batch_size=32,
    validation_split=0.15,
    verbose=1,
    callbacks=[tensorboard_model])
#
print(train_generator.class_indices)
#
model.save(utils.constant.all_states_test_dir() + "rice_model.h5")
