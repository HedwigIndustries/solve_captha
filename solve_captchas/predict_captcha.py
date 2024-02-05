import warnings

import joblib
import numpy as np
from keras.models import load_model

from parse_capthas.parse import extract_essene
from utils import find_contours, resize_letter


def solve_captcha():
    output_folder = '../extracted_captchas'
    captcha = extract_essene(output_folder, essence="captcha")
    grayscale, prepared_contours, distortion = find_contours(captcha)
    if distortion:
        warnings.warn("Captcha have distortion! Maybe wrong answer!", UserWarning)
    predict(grayscale, prepared_contours)


def predict(grayscale, prepared_contours):
    predictions = []
    model = load_model('sequential.keras')

    prepared_contours = sorted(prepared_contours, key=lambda el: el[0])
    for contour in prepared_contours:
        x, y, w, h = contour

        letter = grayscale[y:y + h, x:x + w]
        letter_resized = resize_letter(letter)

        prediction = model.predict(letter_resized)

        class_indices = np.argmax(prediction, axis=1)
        loaded_label_encoder = joblib.load('label_encoder.joblib')
        letter_predict = loaded_label_encoder.inverse_transform(class_indices)
        print(letter_predict)
        predictions.append(letter_predict)
    captcha_solve = "".join(map(str, predictions))
    print(f'CAPTCHA solve is: {captcha_solve}')


def main():
    solve_captcha()


if __name__ == "__main__":
    main()
