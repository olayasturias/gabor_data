#!/usr/bin/env python
# coding: utf-8
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from gabor_utils.parsing_utils import get_args
import re


# this method is a slightly adjusted copy from
# https://stackoverflow.com/questions/19039674/how-can-i-expand-this-gabor-patch-to-the-size-of-the-bounding-box
"""
    lambda_ : int
        Spatial frequency (px per cycle)

    theta : int or float
        Grating orientation in degrees (0-180)

    sigma : int or float
        gaussian standard deviation (in pixels)

    phase : float
        phase offset of the grating, between 0 and 180

    trim : float
        used to cut the gaussian at some point
        preventing it from continuing infinitely
"""


class GaborPatchDataset():
    def __init__(self, image_height=20,
                 n_gabor_patches=1, gabor_patch_ratio=0.8,
                 n_noise_patches=0, output_path="images/",
                 experiment_type="C8"):
        """
        Initialize the GaborPatchDataset class.

        Parameters:
            n_images (int): Number of images to generate.
            n_gabor_patches (int): Number of Gabor patches per image.
            n_noise_patches (int): Number of noise patches per image.
            output_path (str): Path where images and CSV file will be saved.
            image_height (int): Height of each generated image.
        """

        self.n_gabor_patches = n_gabor_patches
        self.n_noise_patches = n_noise_patches
        self.output_path = output_path+experiment_type
        self.image_height = image_height

        # Ensure output directory exists
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        # Define CSV column names
        self.columns = ["image_name"] + [f"orientation_{i}"
                                         for i in range(n_gabor_patches)]  # +\
                                            # [f"shift_{i}"
                                            #  for i in range(n_gabor_patches)]
        # parse experiment type: separate letter from number
        n_rotations, shift = self.set_experiment_type(experiment_type)

        self.n_images = int(n_rotations + shift*n_rotations)

        self.df = pd.DataFrame(columns=self.columns)
        self.generate_dataset_images()

    def set_experiment_type(self, experiment_type):
        match = re.match(r"([a-zA-Z]+)(\d+)", experiment_type)
        if match:
            experiment_type = match.group(1)
            experiment_order = match.group(2)
        else:
            raise ValueError("Invalid experiment_type format")

        if experiment_type == "C":
            print("Cyclic group")
            n_rotations = int(experiment_order)
            shift = 0
        elif experiment_type == "D":
            print("Dihedral group")
        elif experiment_type == "S":
            print("Symmetric group")

        return n_rotations, shift

    def generate_dataset_images(self):
        """
        Generate dataset images with Gabor and noise patches.
        """
        for i in range(self.n_images):
            orientation = i * 360 / self.n_images
            img, orientations = self.generate_dataset_image(
                self.n_gabor_patches,
                self.n_noise_patches, self.image_height, orientation)

            img_name = f"gabor{self.n_gabor_patches}_{i:06d}.png"
            row = [img_name] + orientations

            # Append to DataFrame
            self.df = self.df.append(pd.Series(row, index=self.columns),
                                     ignore_index=True)

            # Save image
            print(f"Saving image {self.output_path}/{img_name}")
            img.save(f"{self.output_path}/{img_name}")

        self.generate_csv()  # Save dataset description to CSV

    def generate_csv(self):
        """
        Save dataset metadata to a CSV file.
        """
        csv_path = f"{self.output_path}/description.csv"
        self.df.to_csv(csv_path, index=False)

    def generate_dataset_image(self,
                               n_gabor_patches=None,
                               n_noise_patches=None,
                               image_height=20,
                               orientation=0):
        patch_size = image_height * 0.5

        img = self.generate_image(image_height)
        if n_gabor_patches:
            img, orientations = self.insert_gabor_patch(
                patch_size, img, orientation)
        if n_noise_patches:
            img = add_noise_patches(img, n_noise_patches, patch_size / 3)
        return img, orientations

    def generate_image(self, image_height):
        background_color = "#7f7f7f"
        total_img = Image.new(
            "L",
            (image_height,
             image_height),
            background_color)

        return total_img

    def insert_gabor_patch(self, patch_size, total_img, orientation):
        lambda_ = 20
        sigma = 0
        orientations = []
        # orientation = np.random.uniform(0, 180)
        orientations.append(orientation)
        phase = 0  # np.random.uniform(0, 360)
        patch = self.gabor_patch(
            int(patch_size),
            lambda_,
            orientation,
            sigma,
            phase,
            binary=True
            )
        total_img.paste(patch,
                        (int((image_height - patch_size) / 2),
                         int((image_height - patch_size) / 2)))
        return total_img, orientations

    def gabor_patch(self, size, lambda_, theta, sigma, phase,
                    trim=.005, binary=False):
        # Create normalized pixel coordinates and create meshgrid
        X = np.linspace(-size//2, size//2, size)
        Y = np.linspace(-size//2, size//2, size)
        Xm, Ym = np.meshgrid(X, Y)

        # Sine wave frequency
        # freq = size / float(lambda_)
        thetaRad = np.deg2rad(theta)
        phaseRad = np.deg2rad(phase)

        # Rotate coordinates to match theta orientation
        Xr = Xm * np.cos(thetaRad) + Ym * np.sin(thetaRad)
        # Create sinusoidal grating with phaseshift phaserad
        grating = np.sin((2 * np.pi * Xr / lambda_) + phaseRad)

        # Convert to black & white (binary)
        if binary:
            grating = np.where(grating >= 0, 1, -1)

        if sigma and sigma > 0:
            # Create gaussian window
            # The gaussian is centered at (0,0) and std is sigma
            gauss = np.exp(
                -((Xm ** 2) + (Ym ** 2)) / (2 * (sigma / float(size)) ** 2)
                )
            gauss[gauss < trim] = 0  # trim values smaller than trim
            grating *= gauss
        # Normalize to range [0,1] and scale to 255
        img_data = (grating + 1) / 2 * 255

        return Image.fromarray(img_data.astype(np.uint8))


def add_noise_patch(img, diameter=50, center=(None, None)):
    diam = round(diameter)
    radius = round(diameter / 2)
    img_width, img_height = img.size

    center_x, center_y = center
    if center_x is None:
        center_x = round(np.random.uniform(0, img_width))
    if center_y is None:
        center_y = round(np.random.uniform(0, img_height))

    h = np.random.uniform(0, 255, (diam, diam))
    s = np.clip((np.random.normal(loc=0.5, scale=0.1, size=(diam, diam))),
                0,
                1) * 255
    v = np.clip((np.random.normal(loc=0.5, scale=0.1, size=(diam, diam))),
                0,
                1) * 255

    start_x = center_x - radius
    start_y = center_y - radius

    for x in range(0, diam):
        for y in range(0, diam):
            coord_x = start_x + x
            coord_y = start_y + y
            if (coord_x > 0 and coord_x < img_width and
                coord_y > 0 and coord_y < img_height and
                    ((x - radius) ** 2 + (y - radius) ** 2) < radius ** 2):
                img.putpixel(
                    (coord_x, coord_y),
                    (int(h[x, y]),
                     int(s[x, y]),
                     int(v[x, y])))
    return img


def add_noise_patches(img, number=5, max_diameter=70, min_diameter_scale=0.8):
    for _ in range(0, number):
        img = add_noise_patch(img,
                              diameter=max_diameter * np.random.uniform(
                                  min_diameter_scale, 1))
    return img


if __name__ == '__main__':
    FLAGS = get_args()
    n_images = FLAGS.n_images
    n_gabor_patches = FLAGS.n_gabor_patches
    n_noise_patches = FLAGS.n_noise_patches
    output_path = FLAGS.output_path
    image_height = FLAGS.image_height

    dataset = GaborPatchDataset(
        n_gabor_patches=n_gabor_patches,
        n_noise_patches=n_noise_patches,
        output_path=output_path,
        image_height=image_height
    )
