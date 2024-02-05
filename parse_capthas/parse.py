import base64
import os
import string
import time
import warnings
from io import BytesIO
from random import choice

from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By

from solve_captchas.utils import find_contours, add_white_pixels


def extract_essene(output_folder, essence="captcha", example_letter_count=100, letter_name=""):
    os.makedirs(output_folder, exist_ok=True)
    chrome = start_parse()
    grammar = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
               'V', 'W', 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    match essence:
        case "letters":
            extract_letters(chrome, grammar, output_folder, example_letter_count, letter_name)
        case "captcha":
            return extract_captcha(chrome, grammar, output_folder)
        case _:
            warnings.warn("Wrong essence! Choose between [\"letters\", \"captcha\"]", UserWarning)


def start_parse():
    webdriver_path = "../parse_capthas/chromedriver.exe"
    url = 'https://captchamaker.com/'
    chrome = run_chrome_driver(webdriver_path)
    chrome.get(url)
    return chrome


def run_chrome_driver(webdriver_path):
    chrome_service = ChromeService(executable_path=webdriver_path)
    chrome_options = webdriver.ChromeOptions()
    return webdriver.Chrome(service=chrome_service, options=chrome_options)


def extract_letters(chrome, grammar, output_folder, example_letter_count, letter_name):
    fill_fields(chrome, length="3")
    for letter in grammar:
        if letter_name == "":
            extract_letter(chrome, example_letter_count, letter, output_folder)
        else:
            extract_letter(chrome, example_letter_count, letter_name, output_folder)
            break
    chrome.quit()


def extract_letter(chrome, example_letter_count, letter, output_folder):
    set_field(chrome, "chars", letter)
    refresh(chrome)
    time.sleep(1)
    os.makedirs(f'{output_folder}/{letter}', exist_ok=True)
    for i in range(example_letter_count):
        try:
            decode_image(chrome, output_folder, f'{letter}/{i}', cropp=True)
        except Exception as e:
            warnings.warn(f"Error occurs: {e.args}")
            chrome.quit()
    print(f'All letters downloaded and saved in the "{letter}" folder.')


def extract_captcha(chrome, grammar, output_folder):
    fill_fields(chrome)
    set_field(chrome, "chars", ''.join(grammar))
    refresh(chrome)
    time.sleep(1)
    try:
        name = generate_random_name(8)
        captcha = decode_image(chrome, output_folder, name)
        print(f'Captcha "{name}" downloaded and saved in the "{output_folder}" folder.')
        return captcha
    except Exception as e:
        warnings.warn(f"Error occurs: {e.args}")
        chrome.quit()
    chrome.quit()


def fill_fields(chrome, width="400", height="100", font_size="40", font_kerning="30", rotation="5", line_count="0",
                line_thickness="1", length="4"):
    set_field(chrome, "width", width)
    set_field(chrome, "height", height)
    set_field(chrome, "font-size", font_size)
    set_field(chrome, "font-kerning", font_kerning)
    set_field(chrome, "rotation", rotation)
    set_field(chrome, "line-count", line_count)
    set_field(chrome, "line-thickness", line_thickness)
    set_field(chrome, "length", length)


def set_field(chrome, _id, value):
    field = chrome.find_element('xpath', f'//input[@id="{_id}"]')
    chrome.execute_script("arguments[0].value='';", field)
    field.send_keys(value)


def refresh(driver):
    button = driver.find_element(By.XPATH, '//button[@id="refresh"]')
    button.click()
    time.sleep(0.2)


def decode_image(chrome, output_folder, name, cropp=False):
    decoded = base64.b64decode(find_image(chrome))
    image = Image.open(BytesIO(decoded))
    if cropp:
        image = cut_letter_by_contour(image)
        refresh(chrome)
    image.save(f'{output_folder}/{name}.png')
    return image


def cut_letter_by_contour(image):
    grayscale, prepared_contours, _ = find_contours(image)
    x, y, w, h = prepared_contours[1]
    letter = grayscale[y:y + h, x:x + w]
    return Image.fromarray(add_white_pixels(letter))


def find_image(driver):
    image = driver.find_element(By.XPATH, '//img[@id="server-preview"]')
    image_url = image.get_attribute('src')
    return image_url.split(',')[1].strip(" ")


def generate_random_name(length=8):
    characters = string.ascii_letters + string.digits
    return ''.join(choice(characters) for _ in range(length))


def main():
    output_folder = '../extracted_letters'
    extract_essene(output_folder, essence="letters", example_letter_count=100)


if __name__ == "__main__":
    main()
