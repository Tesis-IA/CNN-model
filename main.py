# Get training data
import os.path
import tensorflow as tf
import utils.constant

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
