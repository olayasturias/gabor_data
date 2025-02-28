#!/usr/bin/env python
# coding: utf-8
import csv
from PIL import Image, ImageFilter
import numpy as np
import pandas as pd
from pathlib import Path
from gabor_utils.parsing_utils import get_args
import re
import signal
from tqdm import tqdm

# this method is a slightly adjusted copy from
# https://stackoverflow.com/questions/19039674/how-can-i-expand-this-gabor-patch-to-the-size-of-the-bounding-box


class RotatedMNISTDataset():
    def __init__(self, image_height=20, patch_ratio=0.5, n_png_patches=1,
                 n_noise_patches=0, output_path="images/",
                 experiment_type="C8_Z2_5", mnist_path="C:\\Users\\oat\\Documents\\Datasets\\MNIST",
                 sigma=0):

        self.n_png_patches = n_png_patches
        self.n_noise_patches = n_noise_patches
        self.output_path = Path(output_path) / experiment_type
        self.image_height = image_height
        self.patch_ratio = patch_ratio
        self.sigma = sigma

        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        # Parse MNIST data
        n_mnist, self.mnist_data = self.parse_mnist(mnist_path)

        self.n_rotations, self.n_shifts = self.set_experiment_type(experiment_type)
        self.n_images = (self.n_shifts * self.n_rotations) * n_mnist

        self.shift_values = self.generate_CmZn_transforms(self.n_rotations, self.n_shifts, self.image_height // 4)

        self.columns = ["image_name"] + [f"orientation_{i}" for i in range(n_png_patches)]
        if self.n_shifts > 0:
            self.columns += [f"shift_x_{i}" for i in range(n_png_patches)]
            self.columns += [f"shift_y_{i}" for i in range(n_png_patches)]
        self.columns += ["label"]

        self.df = pd.DataFrame(columns=self.columns)
        
        # Open CSV file for appending
        self.csv_path = f"{self.output_path}/description.csv"
        self.csv_file, self.csv_writer, self.existing_images = self.get_processed_images()
        
        existing_imgs = len(self.existing_images)
        generated_imgs =  (self.n_shifts * self.n_rotations) * existing_imgs
        print("found ", generated_imgs, " images generated from ", existing_imgs, "mnist images")
        print("Generating ", self.n_images, " images in total")


        # Track interrupted process
        self.interrupted = False
        signal.signal(signal.SIGINT, self.handle_interrupt)

        self.generate_dataset_images()
        
        # Close CSV file when done
        self.close_csv()

    def handle_interrupt(self, signum, frame):
        """ Handle keyboard interrupt and ensure the CSV is saved. """
        print("\nProcess interrupted. Saving progress...")
        self.close_csv()
        self.interrupted = True

    def parse_mnist(self, mnist_path):
        # read csv files with mnist data
        mnist_train = pd.read_csv(mnist_path + "/mnist_train.csv", header=None)
        mnist_test = pd.read_csv(mnist_path + "/mnist_test.csv", header=None)
        mnist = pd.concat([mnist_train, mnist_test])

        # number of images in mnist:
        n_mnist = mnist.shape[0]
        # parse mnist data
        mnist_images = [Image.fromarray(img.reshape(28, 28).astype(np.uint8)) for img in mnist.iloc[:, 1:].values]
        mnist_labels = mnist.iloc[:, 0].values
        return n_mnist, (mnist_images, mnist_labels)

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
        """ Generate dataset images and resume if interrupted. """
        mnist_images, mnist_labels = self.mnist_data
        generated_imgs_size = len(self.existing_images)
        for idx, (mnist_image, mnist_label) in tqdm(enumerate(zip(mnist_images, mnist_labels)), total=len(mnist_images)):
            if idx < generated_imgs_size:
                continue  # Skip already processed MNIST rows

            for i, (orientation, shift_x, shift_y) in enumerate(self.shift_values):
                if self.interrupted:
                    return  # Stop processing immediately if interrupted

                img, orientations, shifts = self.generate_dataset_image(
                    mnist_image, self.n_noise_patches, self.image_height,
                    orientation, shift_x, shift_y, self.sigma
                )

                img_name = f"mnist_digit_{mnist_label}_img_{idx:06d}_transformation_{i:03d}.png"
                img_path = Path(self.output_path) / str(mnist_label)
                img_path.mkdir(parents=True, exist_ok=True)
                img_file = Path(img_path)/img_name
                img_rel_path = str(mnist_label)+"/"+img_name
                row = [img_rel_path] + orientations
                if self.n_shifts > 0:
                    row += shifts
                row += [mnist_label]


                # Append new row to CSV file
                self.csv_writer.writerow(row)
                self.csv_file.flush()

                # print(f"Saving image {self.output_path}/{img_name}")
                img.save(img_file)

    def get_processed_images(self):
        """ Open description.csv in append mode and read processed images. """
        csv_path = self.csv_path
        processed_indices = set()

        # Open the CSV file in append mode
        file = open(csv_path, 'a', newline='') if Path(csv_path).exists() else open(csv_path, 'w', newline='')
        writer = csv.writer(file)

        # Read existing data to track processed indices
        if Path(csv_path).exists():
            with open(csv_path, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader, None)  # Skip header if it exists
                for row in reader:
                    match = re.search(r"img_(\d+)", row[0])
                    if match:
                        processed_indices.add(int(match.group(1)))

        # If new file, write header
        if not processed_indices:
            writer.writerow(self.columns)

        return file, writer, processed_indices

    def generate_csv(self):
        """ Save dataset metadata to a CSV file. """
        csv_path = f"{self.output_path}/description.csv"
        self.df.to_csv(csv_path, index=False)
        
    def close_csv(self):
        """ Close the CSV file safely. """
        if self.csv_file:
            self.csv_file.close()
            print("CSV file closed.")

    def generate_dataset_image(self,
                               png_patch=None,
                               n_noise_patches=None,
                               image_height=20,
                               orientation=0,
                               shift_x=0,
                               shift_y=0,
                               sigma=0
                               ):
        patch_size = image_height * self.patch_ratio

        img = self.generate_image(image_height)
        if png_patch:
            img, orientations, shifts = self.insert_png_patch(
                png_patch, patch_size, img, orientation, shift_x, shift_y)

        if n_noise_patches:
            img = add_noise_patches(img, n_noise_patches, patch_size / n_noise_patches)
        return img, orientations, shifts

    def generate_image(self, image_height):
        background_color = "black"
        total_img = Image.new(
            "L",
            (image_height,
             image_height),
            background_color)

        return total_img

    def insert_png_patch(self, patch, patch_size, total_img, orientation, shift_x=0, shift_y=0):
        # Resize patch
        patch = patch.resize((int(patch_size), int(patch_size)))

        # Rotate the patch and its alpha mask
        patch_rotated = patch.rotate(orientation, expand=True)

        # Debug: Show rotated patch
        # import matplotlib.pyplot as plt
        # plt.imshow(patch_rotated, cmap='gray')
        # plt.title(f'PNG Patch (Rotated {orientation}°)')
        # plt.axis('off')
        # plt.show()

        # Compute paste position
        center_x = (self.image_height - patch_rotated.width) // 2 + shift_x
        center_y = (self.image_height - patch_rotated.height) // 2 + shift_y

        # Ensure patch stays within bounds
        center_x = int(max(0, min(center_x, self.image_height - patch_rotated.width*0.7)))
        center_y = int(max(0, min(center_y, self.image_height - patch_rotated.height*0.7)))

        # Paste the rotated patch onto the total image using transparency mask
        total_img.paste(patch_rotated, (center_x, center_y))

        return total_img, [orientation], [shift_x, shift_y]

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
    patch_ratio = FLAGS.patch_ratio
    n_noise_patches = FLAGS.n_noise_patches
    output_path = FLAGS.output_path
    experiment_type = FLAGS.experiment_type
    sigma = FLAGS.sigma
    n_png_patches = FLAGS.n_png_patches

    dataset = RotatedMNISTDataset(
        image_height=image_height,
        n_png_patches=n_png_patches,
        patch_ratio=patch_ratio,
        n_noise_patches=n_noise_patches,
        output_path=output_path,
        experiment_type=experiment_type,
        sigma=sigma
    )
