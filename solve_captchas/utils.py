import os

import cv2
import joblib
import numpy as np
from imutils.paths import list_images
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder


def find_contours(captcha):
    captcha_image = np.array(captcha, dtype='uint8')
    grayscale = cv2.cvtColor(captcha_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grayscale, 0, 250, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    prepared_contours = []
    distortion = False
    for contour in contours:
        rect_contours = cv2.boundingRect(contour)
        x, y, w, h = rect_contours

        if h < 25:
            distortion = True
            continue

        prepared_contours.append(rect_contours)

    return grayscale, prepared_contours, distortion


def resize_letter(letter):
    letter_resized = add_white_pixels(letter)
    letter_resized = cv2.resize(letter_resized, (50, 50))
    return np.expand_dims(letter_resized, axis=0)


def add_white_pixels(letter):
    bottom_padding, left_padding, right_padding, top_padding = calc_padding(letter)
    letter_resized = cv2.copyMakeBorder(letter, top_padding, bottom_padding, left_padding, right_padding,
                                        cv2.BORDER_CONSTANT, value=[255, 255, 255])
    return letter_resized


def calc_padding(letter):
    desired_height, desired_width = max(50, letter.shape[0]), max(50, letter.shape[1])
    height_diff = desired_height - letter.shape[0]
    width_diff = desired_width - letter.shape[1]
    top_padding = height_diff // 2
    bottom_padding = height_diff - top_padding
    left_padding = width_diff // 2
    right_padding = width_diff - left_padding
    return bottom_padding, left_padding, right_padding, top_padding


def read_data(path_to_letters):
    letters = []
    labels = []
    for image_file in list_images(path_to_letters):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = image_file.split(os.path.sep)[1]
        letters.append(image)
        labels.append(label)
    return letters, labels


def prepare_data(labels, letters):
    letters = [cv2.resize(img, (50, 50)) for img in letters]
    letters = np.array(letters, dtype="float32") / 255
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    one_hot_labels = to_categorical(encoded_labels)
    joblib.dump(label_encoder, 'label_encoder.joblib')
    return letters, one_hot_labels, len(label_encoder.classes_)
