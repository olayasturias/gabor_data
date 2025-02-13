#!/usr/bin/env python
# coding: utf-8
from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
from pathlib import Path
from gabor_utils.parsing_utils import get_args
import re


# this method is a slightly adjusted copy from
# https://stackoverflow.com/questions/19039674/how-can-i-expand-this-gabor-patch-to-the-size-of-the-bounding-box


class GaborPatchDataset():
    """
    A class to generate a dataset of images containing Gabor patches
    and noise patches.

    Attributes
    ----------
    n_gabor_patches : int
        Number of Gabor patches per image.
    n_noise_patches : int
        Number of noise patches per image.
    output_path : str
        Path where images and CSV file will be saved.
    image_height : int
        Height of each generated image.
    gabor_patch_ratio : float
        Ratio of the Gabor patch size to the image height.
    n_rotations : int
        Number of rotations for the experiment.
    n_shifts : int
        Number of shifts for the experiment.
    n_images : int
        Total number of images to generate.
    shift_values : list
        List of shift values for the experiment.
    columns : list
        List of column names for the CSV file.
    df : pandas.DataFrame
        DataFrame to store metadata of the generated images.

    Methods
    -------
    set_experiment_type(experiment_type)
        Parse the experiment type to determine rotations and shifts.
    generate_dataset_images()
        Generate dataset images with Gabor and noise patches.
    generate_csv()
        Save dataset metadata to a CSV file.
    generate_dataset_image(n_gabor_patches, n_noise_patches,
        image_height, orientation, shift_x, shift_y)
        Generate a single dataset image with Gabor and noise patches.
    generate_image(image_height)
        Generate a blank image with a specified height.
    insert_gabor_patch(patch_size, total_img, orientation, shift_x, shift_y)
        Insert a Gabor patch into an image.
    gabor_patch(size, lambda_, theta, sigma, phase, trim, binary)
        Create a Gabor patch.
    generate_random_shifts(n_shifts, image_height)
        Generate random shifts for the Gabor patches.
    generate_CmZn_transforms(m_rotations, n_shifts, shift_distance)
        Generate transformations for the Gabor patches
            based on rotations and shifts.
    """
    def __init__(self, image_height=20,
                 n_gabor_patches=1, gabor_patch_ratio=0.8,
                 n_noise_patches=0, output_path="images/",
                 experiment_type="C8_Z2_5",  # C8_Z2_5, C4
                 sigma=0,
                 ):

        self.n_gabor_patches = n_gabor_patches
        self.n_noise_patches = n_noise_patches
        self.output_path = Path(output_path) / experiment_type
        self.image_height = image_height
        self.gabor_patch_ratio = gabor_patch_ratio
        self.sigma = sigma

        # Ensure output directory exists
        Path(self.output_path).mkdir(parents=True, exist_ok=True)
        # print output path
        print(f"Output path: {self.output_path}")

        # parse experiment type to determine rotations and shifts
        self.n_rotations, self.n_shifts = \
            self.set_experiment_type(experiment_type)
        self.n_images = self.n_rotations + self.n_shifts * self.n_rotations

        self.shift_values = self.generate_CmZn_transforms(
            self.n_rotations, self.n_shifts, self.image_height // 4)

        # Define CSV column names
        self.columns = ["image_name"] + [f"orientation_{i}"
                                         for i in range(n_gabor_patches)]
        if self.n_shifts > 0:
            self.columns += [f"shift_x_{i}" for i in range(n_gabor_patches)]
            self.columns += [f"shift_y_{i}" for i in range(n_gabor_patches)]

        self.df = pd.DataFrame(columns=self.columns)
        self.generate_dataset_images()

    def set_experiment_type(self, experiment_type):
        match = re.match(r"C(\d+)(_Z2_(\d+))?", experiment_type)
        if match:
            rotations = int(match.group(1))  # Extract number of rotations
            # Extract number of translations, default to 0
            translations = int(match.group(3)) if match.group(3) else 0
            print(f"Rotations: {rotations}, Translations: {translations}")
            return rotations, translations
        else:
            raise ValueError("Invalid experiment_type format. \
                             Expected format: 'C<number>_Z2_<number>'")

    def generate_dataset_images(self):
        """
        Generate dataset images with Gabor and noise patches.
        """
        # for i in range(self.n_images):
        for i, (orientation, shift_x, shift_y) in enumerate(self.shift_values):
            img, orientations, shifts = self.generate_dataset_image(
                self.n_gabor_patches,
                self.n_noise_patches,
                self.image_height,
                orientation,
                shift_x,
                shift_y,
                self.sigma
                )

            img_name = f"gabor{self.n_gabor_patches}_{i:06d}.png"
            row = [img_name] + orientations
            if self.n_shifts > 0:
                row += shifts

            # Append to DataFrame
            self.df = pd.concat([self.df, pd.Series(row, index=self.columns).to_frame().T], ignore_index=True)

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
                               orientation=0,
                               shift_x=0,
                               shift_y=0,
                               sigma=0
                               ):
        patch_size = image_height * self.gabor_patch_ratio

        img = self.generate_image(image_height)
        if n_gabor_patches:
            img, orientations, shifts = self.insert_gabor_patch(
                patch_size, img, orientation, shift_x, shift_y,
                sigma
            )
        if n_noise_patches:
            img = add_noise_patches(img, n_noise_patches, patch_size / n_noise_patches)
        return img, orientations, shifts

    def generate_image(self, image_height):
        background_color = "#7f7f7f"
        total_img = Image.new(
            "L",
            (image_height,
             image_height),
            background_color)

        return total_img

    def insert_gabor_patch(self, patch_size, total_img,
                           orientation, shift_x=0, shift_y=0,
                           sigma=0):
        lambda_ = 20
        sigma = sigma
        if sigma:
            binary = False
        else:
            binary = True
        orientations = []
        shifts = [shift_x, shift_y]
        # orientation = np.random.uniform(0, 180)
        orientations.append(orientation)
        phase = 0  # np.random.uniform(0, 360)
        patch = self.gabor_patch(
            int(patch_size),
            lambda_,
            orientation,
            phase,
            binary=binary
            )
        center_x = (self.image_height - patch_size) // 2 + shift_x
        center_y = (self.image_height - patch_size) // 2 + shift_y
        # Ensure the patch stays within the image boundaries
        center_x = int(max(0, min(center_x, self.image_height - patch_size)))
        center_y = int(max(0, min(center_y, self.image_height - patch_size)))
        total_img.paste(patch, (center_x, center_y))
        if sigma and sigma > 0:
            kernel_size = int(sigma) | 1  # Make sure it's odd
            total_img = total_img.filter(ImageFilter.GaussianBlur(radius=kernel_size))

        return total_img, orientations, shifts

    def gabor_patch(self, size, lambda_, theta, phase,
                    trim=.005, binary=False):
        # Create normalized pixel coordinates and create meshgrid
        X = np.linspace(-size//2, size//2, size)
        Y = np.linspace(-size//2, size//2, size)
        Xm, Ym = np.meshgrid(X, Y)

        # Convert angles
        thetaRad = np.deg2rad(theta)
        phaseRad = np.deg2rad(phase)

        # Rotate coordinates
        Xr = Xm * np.cos(thetaRad) + Ym * np.sin(thetaRad)

        # Create sinusoidal grating
        grating = np.sin((2 * np.pi * Xr / lambda_) + phaseRad)

        # Convert to black & white (binary)
        if binary:
            grating = np.where(grating >= 0, 1, -1)

        # Create a circular mask
        radius = size // 2  # Radius of the circular mask
        mask = (Xm**2 + Ym**2) <= radius**2  # Circle equation x² + y² <= r²

        # Apply the circular mask
        img_data = np.full_like(grating, 127)  # Background (gray)
        img_data[mask] = (grating[mask] + 1) / 2 * 255  # Apply Gabor pattern inside the circle

        return Image.fromarray(img_data.astype(np.uint8))

    def generate_random_shifts(self, n_shifts, image_height):
        shifts = []
        if n_shifts == 0:
            return [(0, 0)]  # No shifts, only center position
        for _ in range(n_shifts):
            shift_x = np.random.randint(-image_height // 4, image_height // 4)
            shift_y = np.random.randint(-image_height // 4, image_height // 4)
            shifts.append((shift_x, shift_y))
        return shifts

    def generate_CmZn_transforms(self, m_rotations, n_shifts, shift_distance):
        if n_shifts == 0:  # No shift, just angles
            return [(angle, 0, 0) for angle in
                    np.linspace(0, 360, m_rotations, endpoint=False)]

        all_shifts = []

        for rot_idx in range(m_rotations):
            rotation_angle = (rot_idx * 360 / m_rotations)  # Orientation angle

            # Always include the (0,0) shift
            all_shifts.append((rotation_angle, 0, 0))

            # Generate the remaining shifts uniformly distributed in a circle
            for shift_idx in range(n_shifts-1):
                # Evenly distribute shifts around the circle
                shift_angle = (shift_idx * 360 / (n_shifts-1))
                rad = np.deg2rad(shift_angle)
                shift_x = int(np.round(shift_distance * np.cos(rad)))
                shift_y = int(np.round(shift_distance * np.sin(rad)))
                # Append each shift with rotation
                all_shifts.append((rotation_angle, shift_x, shift_y))

        return all_shifts


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
                coord = (coord_x, coord_y)
                value = (int(h[x, y]), int(s[x, y]), int(v[x, y]))
                img.putpixel(
                    (coord_x, coord_y),
                    int(h[x, y]))
    return img


def add_noise_patches(img, number=5, max_diameter=70, min_diameter_scale=0.8):
    for _ in range(0, number):
        img = add_noise_patch(img,
                              diameter=max_diameter * np.random.uniform(
                                  min_diameter_scale, 1))
    return img


if __name__ == '__main__':
    FLAGS = get_args()
    image_height = FLAGS.image_height
    n_gabor_patches = FLAGS.n_gabor_patches
    gabor_patch_ratio = FLAGS.gabor_patch_ratio
    n_noise_patches = FLAGS.n_noise_patches
    output_path = FLAGS.output_path
    experiment_type = FLAGS.experiment_type
    sigma = FLAGS.sigma

    dataset = GaborPatchDataset(
        image_height=image_height,
        gabor_patch_ratio=gabor_patch_ratio,
        n_gabor_patches=n_gabor_patches,
        n_noise_patches=n_noise_patches,
        output_path=output_path,
        experiment_type=experiment_type,
        sigma=sigma
    )
