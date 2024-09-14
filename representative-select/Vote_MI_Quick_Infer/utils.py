# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import mnist2.mnist as mn
from matplotlib.colors import ListedColormap
import pandas as pd

def prepare_dataset(args):    
    # prepare dataset
    if args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        #(x_train, y_train), (x_test, y_test) = mn.read_data_sets(r"C:\Users\ywang687\Downloads\Brain_tumor_nii\test\convert_MNIST", one_hot=True, num_classes=3)
    elif args.dataset == 'fashion-mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    input_shape = x_train[0].shape[0] * x_train[0].shape[1]
    
    # reshape
    x_train = x_train.reshape(-1, input_shape)
    x_test = x_test.reshape(-1, input_shape)
    
    # one-hot encoding
    y_train = tf.one_hot(y_train, depth=10)
    y_test = tf.one_hot(y_test, depth=10)
    
    # normalization
    x_train = x_train / 255.
    x_test = x_test / 255.
    
    return x_train, y_train, x_test, y_test, input_shape

class Plot_Reproduce_Result():
    def __init__(self, output_dir, n_img_x=8, n_img_y=8, img_w=28, img_h=28):
        self.dir = output_dir
        
        assert n_img_x > 0 and n_img_y > 0
        
        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_total = n_img_x * n_img_y
        
        assert img_w > 0 and img_h > 0
        
        self.img_w = img_w
        self.img_h = img_h
    
    def save_image(self, images, name='result.jpg'):
        images = images.reshape((-1, self.img_h, self.img_w))
        merged_img = self._merge(images)
        merged_img *= 255
        cv2.imwrite(self.dir + '/' + name, merged_img)
    
    def _merge(self, images):
        img = np.zeros((self.img_h * self.n_img_y, self.img_w * self.n_img_x))
        
        for idx, image in enumerate(images):
            i = int(idx / self.n_img_x)
            j = int(idx % self.n_img_x)
            
            img[i * self.img_h:i * self.img_h + self.img_h, 
                j * self.img_w:j * self.img_w + self.img_w] = image
        
        return img

class Plot_Manifold_Learning_Result():
    def __init__(self, output_dir, n_img_x=20, n_img_y=20, img_w=28, img_h=28, z_range=4):
        self.dir = output_dir
        
        assert n_img_x > 0 and n_img_y > 0
        
        self.n_img_x = n_img_x
        self.n_img_y = n_img_y
        self.n_total = n_img_x * n_img_y
        
        assert img_w > 0 and img_h > 0
        
        self.img_w = img_w
        self.img_h = img_h
        
        assert z_range > 0
        self.z_range = z_range
        
        self._set_latent_vectors()
        
    def _set_latent_vectors(self):
        z = np.rollaxis(np.mgrid[self.z_range:-self.z_range:self.n_img_y * 1j, self.z_range:-self.z_range:self.n_img_x * 1j], 0, 3)
        self.z = z.reshape([-1, 2])
    
    def save_image(self, images, name='result.jpg'):
        images = images.reshape((-1, self.img_h, self.img_w))
        merged_img = self._merge(images)
        merged_img *= 255
        cv2.imwrite(self.dir + '/' + name, merged_img)
    
    def _merge(self, images):
        img = np.zeros((self.img_h * self.n_img_y, self.img_w * self.n_img_x))
        
        for idx, image in enumerate(images):
            i = int(idx / self.n_img_x)
            j = int(idx % self.n_img_x)
            
            img[i * self.img_h:i * self.img_h + self.img_h, 
                j * self.img_w:j * self.img_w + self.img_w] = image
        
        return img
    
    def save_scattered_image(self, z, id, name='scattered_image.pdf'):
        N = 10
        plt.figure(figsize=(8, 6))
        #plt.scatter(z[:, 0], z[:, 1], c='dodgerblue', marker='o', edgecolor='none')
        #change N to change the color categories
        plt.scatter(z[:, 0], z[:, 1], c=np.argmax(id, 1), marker='o', edgecolor='none', cmap=discrete_cmap(10, 'jet'))
        plt.colorbar(ticks=range(N))
        axes = plt.gca()
        axes.set_xlim([-self.z_range-2, self.z_range+2])
        axes.set_ylim([-self.z_range-2, self.z_range+2])
        plt.grid(True)
        plt.savefig(self.dir + "/" + name)

    def save_scattered_image_csv(self, z, id, name='scattered_image.csv'):
        N = 10
        z1 = z[:, 0]
        z2 = z[:, 1]
        categories = np.argmax(id, 1)
        df = pd.DataFrame({'z1': z1, 'z2': z2, 'category': categories})
        df.to_csv(self.dir + '/' + name, index=False)

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


# class MCMCSampler:
#     def __init__(self, step_size=0.1, n_samples=1000, burn_in=200):
#         self.step_size = step_size
#         self.n_samples = n_samples
#         self.burn_in = burn_in
#
#     def sample(self, data, n_samples):
#         """Performs MCMC sampling.
#
#         Args:
#             data (np.ndarray): The latent space points to sample from.
#             n_samples (int): The number of samples to generate.
#
#         Returns:
#             np.ndarray: The sampled points.
#         """
#         # Initialize samples
#         current_sample = data[np.random.choice(len(data))]
#         samples = [current_sample]
#
#         for _ in range(self.n_samples + self.burn_in):
#             # Propose a new sample
#             proposal = self.propose_new_sample(current_sample)
#
#             # Compute acceptance probability
#             acceptance_prob = self.acceptance_probability(current_sample, proposal, data)
#
#             # Accept or reject the proposal
#             if np.random.rand() < acceptance_prob:
#                 current_sample = proposal
#
#             samples.append(current_sample)
#
#         # Discard burn-in samples and return the desired number of samples
#         return np.array(samples[self.burn_in:self.burn_in + n_samples])
#
#     def propose_new_sample(self, current_sample):
#         """Proposes a new sample based on the current sample.
#
#         Args:
#             current_sample (np.ndarray): The current sample in the MCMC chain.
#
#         Returns:
#             np.ndarray: The proposed new sample.
#         """
#         # Propose a new point by adding a small random perturbation
#         return current_sample + np.random.normal(0, self.step_size, size=current_sample.shape)
#
#     def acceptance_probability(self, current_sample, proposal, data):
#         """Calculates the acceptance probability.
#
#         Args:
#             current_sample (np.ndarray): The current sample.
#             proposal (np.ndarray): The proposed new sample.
#             data (np.ndarray): The latent space points.
#
#         Returns:
#             float: The acceptance probability.
#         """
#         # For a simple implementation, we assume a uniform prior and calculate based on the distance
#         current_density = self.target_density(current_sample, data)
#         proposal_density = self.target_density(proposal, data)
#
#         # Compute the acceptance probability (assuming a symmetric proposal distribution)
#         return min(1, proposal_density / current_density)
#
#     def target_density(self, sample, data):
#         """Calculates the target density function, which in this case could be
#         the proximity to other points in the dataset.
#
#         Args:
#             sample (np.ndarray): The current sample.
#             data (np.ndarray): The latent space points.
#
#         Returns:
#             float: The target density value.
#         """
#         # Calculate the density based on the proximity to other points in the data
#         # For simplicity, using a Gaussian kernel density estimate here
#         distances = np.linalg.norm(data - sample, axis=1)
#         density = np.sum(np.exp(-distances ** 2 / (2 * self.step_size ** 2)))
#         return density