import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
from glob import glob
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from unet import build_unet
from metrics import categorical_dice_loss, categorical_dice_coef

""" Global parameters """
H = 640
W = 640
num_classes = 3  # Change this to the number of classes in your multiclass segmentation task

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path, split=0.2):
    images = sorted(glob(os.path.join(path, "images", "*.jpg")))
    masks = sorted(glob(os.path.join(path, "masks", "*.png")))

    split_size = int(len(images) * split)

    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=split_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=split_size, random_state=42)

    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (W, H))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

import numpy as np
import cv2

def read_mask(path, num_classes):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # Read mask as grayscale
    x = cv2.resize(x, (W, H))   # Resize if necessary
    x = x / 255.0   # Normalize to [0, 1] range
    x = x.astype(np.float32)   # Convert to float32
    # Perform one-hot encoding
    masks = []
    for class_label in range(num_classes):
        mask = np.where(x == class_label, 1.0, 0.0)  # Binary mask for each class
        masks.append(mask)
    # Stack binary masks along the channel axis to create one-hot encoded mask
    one_hot_mask = np.stack(masks, axis=-1)
    return one_hot_mask.astype(np.float32)  # Ensure all values are float32



def tf_parse(x, y, num_classes):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y, num_classes)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, num_classes])  # Update to match the number of classes
    return x, y


def tf_dataset(X, Y, num_classes, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(lambda x, y: tf_parse(x, y, num_classes))  # Pass num_classes here
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset


if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("files")

    """ Hyperparameters """
    batch_size = 6
    lr = 1e-4
    num_epochs = 100
    model_path = os.path.join("files", "model.h5")
    csv_path = os.path.join("files", "log.csv")

    """ Dataset """
    dataset_path = r"C:\Users\abhad\OneDrive\Pictures\finalone"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    print(f"Train: {len(train_x)} - {len(train_y)}")
    print(f"Valid: {len(valid_x)} - {len(valid_y)}")
    print(f"Test : {len(test_x)} - {len(test_y)}")

    train_dataset = tf_dataset(train_x, train_y, num_classes, batch=batch_size)
    valid_dataset = tf_dataset(valid_x, valid_y, num_classes, batch=batch_size)

    """ Model """
    model = build_unet((H, W, 3), num_classes)
    model.compile(loss=categorical_dice_loss, optimizer=Adam(lr), metrics=[categorical_dice_coef])

    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]

    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )
