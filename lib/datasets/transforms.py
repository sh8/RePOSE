import numpy as np
import random
from torchvision import transforms
import cv2
from PIL import Image


class RandomBlur(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            sigma = np.random.choice([3, 5, 7, 9])
            image = cv2.GaussianBlur(image, (sigma, sigma), 0)
        return image


def make_transforms(cfg, is_train):
    if is_train is True:
        transform = transforms.Compose([
            RandomBlur(0.5),
            transforms.ToTensor(),
            transforms.ColorJitter(0.1, 0.1, 0.05, 0.05),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    return transform
