import os
import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from metrics import categorical_dice_loss, categorical_dice_coef
from train import load_dataset, create_dir
from sklearn.metrics import confusion_matrix


H = 640
W = 640

def remap_mask(mask):
    mask[mask == 29] = 1
    mask[mask == 76] = 2
    return mask
""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_results(image, mask, y_pred, save_image_path):

    # Convert mask and prediction to the same size images as the original input for visualization
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    y_pred_argmax = np.argmax(y_pred, axis=-1)
    y_pred_argmax = cv2.resize(y_pred_argmax, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    insulator_class_index = 1
    unhealthy_class_index = 2
    overlay_image = image.copy()
    overlay_image[y_pred_argmax == insulator_class_index] = [255, 0, 0]  # Red for insulator
    overlay_image[y_pred_argmax == unhealthy_class_index] = [0, 0, 255]  # Blue for unhealthy parts

    # Create a side by side comparison image
    line = np.ones((image.shape[0], 10, 3), dtype=np.uint8) * 255
    comparison_image = np.concatenate([image, line, overlay_image], axis=1)

    # Save the comparison image
    cv2.imwrite(save_image_path, comparison_image)

if __name__ == "__main__":
    true_labels = []
    pred_labels = []
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Load the model """
    with CustomObjectScope({"categorical_dice_coef": categorical_dice_coef, "categorical_dice_loss": categorical_dice_loss}):
        model = tf.keras.models.load_model(os.path.join("files", "model.h5"))

    """ Dataset """
    dataset_path = r"D:\Division\type1-Data"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
    mean_iou = MeanIoU(num_classes=3)
    SCORE = []

    for x, y in tqdm(zip(test_x, test_y), total=len(test_y)):
        name = x.split("/")[-1]
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (W, H))
        x = image / 255.0
        x = np.expand_dims(x, axis=0)

        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (W, H))
        mask = remap_mask(mask)  # Remap the mask values

        y_pred = model.predict(x)[0]
        save_image_path = os.path.join("results", name)
        save_results(image, mask, y_pred, save_image_path)

        mask_flat = mask.flatten()
        y_pred_flat = np.argmax(y_pred, axis=-1).flatten()
        true_labels.append(mask_flat)
        pred_labels.append(y_pred_flat)


        mean_iou.update_state(mask_flat, y_pred_flat)
        """ Calculating the metrics values """
        f1_value = f1_score(mask_flat, y_pred_flat, labels=[0, 1, 2], average="weighted")
        jac_value = jaccard_score(mask_flat, y_pred_flat, labels=[0, 1, 2], average="weighted")
        recall_value = recall_score(mask_flat, y_pred_flat, labels=[0, 1, 2], average="weighted", zero_division=0)
        precision_value = precision_score(mask_flat, y_pred_flat, labels=[0, 1, 2], average="weighted", zero_division=0)
        SCORE.append([name, f1_value, jac_value, recall_value, precision_value])
    true_labels = np.concatenate(true_labels)
    pred_labels = np.concatenate(pred_labels)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    print("Confusion Matrix:")
    print(cm)
    final_mean_iou = mean_iou.result().numpy()
    print(f"Mean IoU: {final_mean_iou}")

    """ Metrics values """
    score = [s[1:]for s in SCORE]
    score = np.mean(score, axis=0)
    print(f"Weighted F1: {score[0]:0.5f}")
    print(f"Weighted Jaccard: {score[1]:0.5f}")
    print(f"Weighted Recall: {score[2]:0.5f}")
    print(f"Weighted Precision: {score[3]:0.5f}")

    df = pd.DataFrame(SCORE, columns=["Image", "Weighted F1", "Weighted Jaccard", "Weighted Recall", "Weighted Precision"])
    df.to_csv("files/score.csv")

