import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_gabor_patches', type=int, default=3,
                        help="The number of gabor patches on the image")
    parser.add_argument('--n_noise_patches', type=int, default=24,
                        help="The number of gaussian noise patches \
                            in the image")
    parser.add_argument('--color_noise', type=float, default=0,
                        help="Standard deviation of the gaussian \
                            to add noise to the gabor patches (in degrees)")
    parser.add_argument('--image_height', type=int, default=250,
                        help='Height of the image. \
                        The width will be n_gabor_patches time the height')
    parser.add_argument('--output_path', type=str, default='images/',
                        help="Location for saving the images")
    parser.add_argument('--n_images', type=int, default=100,
                        help="The number of images generated")
    parser.add_argument('--gabor_patch_ratio', type=float, default=0.5,
                        help="The ratio of the gabor patches in the image")
    parser.add_argument('--experiment_type', type=str, default='C8',
                        help="The type of experiment to run. \
                        Options: Cx, Cx_Z2_y")
    return parser.parse_args()
