"""
-- Created by Shreejan Shrestha
-- Treeleaf Technologies Pvt. Ltd.
-- Created on 2023-05-30
"""

import os
import random
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageOps
import numpy.typing as npt
from krakenOCR.kraken_ocr import KrakenOCR


def visualize_polylines(image: npt.NDArray, lines: List) -> npt.NDArray:
    """

    draws a polyline round the detected texts.

    :param image: image read with cv2 to draw the polylines. PIL read image is not accepted
    :param lines:
    :return:
    """
    if lines:
        for boundary in lines:
            boundary = np.array(boundary, np.int32)
            r = random.randint(0, 256)
            g = random.randint(0, 256)
            b = random.randint(0, 256)
            image = cv2.polylines(image, [boundary],
                                  False, (r, g, b), 4)

        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    else:
        cv2.namedWindow(f'img', cv2.WINDOW_NORMAL)
        cv2.imshow(f'img', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print('Could not detect anything')

    return image


def multiple_images_experiment(dir_pth: npt.NDArray, lines: List):
    """
    Example-run on multiple images.
    """
    img_dir_path = sorted(os.listdir(dir_pth))
    for img in img_dir_path:
        img_path = os.path.join(dir_pth, img)
        print('Image name', img_path)
        img = cv2.imread(img_path)
        image = Image.open(img_path)
        image = ImageOps.exif_transpose(image)

        kseg = KrakenOCR()

        dt_boxes_b = kseg.ocr(image)

        visualize_polylines(img, dt_boxes_b)


def single_image_experiment(img_path: str):
    """
    Example-run on a single image.
    """
    img = cv2.imread(img_path)
    image = Image.open(img_path)
    image = ImageOps.exif_transpose(image)

    kseg = KrakenOCR()

    dt_boxes = kseg.ocr(image)

    dt_boxes = [i['boundary'] for i in dt_boxes]
    # instead of 4 cornor points, boundary will have polygon points.

    visualize_polylines(img, dt_boxes)


if __name__ == '__main__':
    img_path = '/home/dell/Documents/handwritten_images/testingimages/d1.jpeg'
    single_image_experiment(img_path)
    # dir_path = 'path/to/the/directory/of/images'
    # multiple_images_experiment(dir_path)
