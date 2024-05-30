
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import cv2
import pandas as pd
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from train import load_dataset

smooth = 1e-15
def dice_coef(y_true, y_pred):
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

""" Global parameters """
H = 256
W = 256

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_results(image, mask, y_pred, save_image_path):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    
    # convert the mask to a 2D array
    mask = mask[:, :, 0]
    mask = mask * 255

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    # convert the mask to a 2D array
    y_pred = y_pred[:, :, 0]
    y_pred = y_pred * 255

    line = np.ones((H, 10)) * 255  # Changed this line

    cat_images = np.concatenate([image, line, mask, line], axis=1)  # Changed axis to 1
    cv2.imwrite(save_image_path, cat_images)


if __name__ == "__main__":
    print("[INFO] Testing the model...")
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    print("[INFO] Creating directories...")
    """ Directory for storing files """
    create_dir("results")


    print("[INFO] Loading the model...")
    """ Load the model """
    with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
        model = tf.keras.models.load_model(os.path.join("files", "model.h5"))

    print("[INFO] Loading the dataset...")
    """ Dataset """
    dataset_path = "dataset"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)

    print("[INFO] Predicting the results...")
    """ Prediction and Evaluation """
    SCORE = []
    for x, y in tqdm(zip(test_x, test_y), total=len(test_y)):
        """ Extracting the name """
        name = x.split("/")[-1]

        """ Reading the image """
        image = cv2.imread(x, cv2.IMREAD_COLOR) ## [H, w, 3]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) ## Convert to grayscale
        image = cv2.resize(image, (W, H))       ## [H, w]
        x = image/255.0                         ## [H, w]
        x = np.expand_dims(x, axis=0)           ## [1, H, w]
        x = np.expand_dims(x, axis=-1)

        """ Reading the mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  # This will load the image as grayscale
        mask = cv2.resize(mask, (W, H))

        """ Prediction """
        y_pred = model.predict(x, verbose=0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred >= 0.5
        y_pred = y_pred.astype(np.int32)

        """ Saving the prediction """
        save_image_path = os.path.join("results", name)
        save_results(image, mask, y_pred, save_image_path)

        """ Flatten the array """
        mask = mask/255.0
        mask = (mask > 0.5).astype(np.int32).flatten()
        y_pred = y_pred.flatten()

        """ Calculating the metrics values """
        f1_value = f1_score(mask, y_pred, labels=[0, 1], average="binary")
        jac_value = jaccard_score(mask, y_pred, labels=[0, 1], average="binary")
        recall_value = recall_score(mask, y_pred, labels=[0, 1], average="binary", zero_division=0)
        precision_value = precision_score(mask, y_pred, labels=[0, 1], average="binary", zero_division=0)
        SCORE.append([name, f1_value, jac_value, recall_value, precision_value])

    print("[INFO] Saving the results...")
    """ Metrics values """
    score = [s[1:]for s in SCORE]
    score = np.mean(score, axis=0)

    print("[INFO] Prediciton Done...")
