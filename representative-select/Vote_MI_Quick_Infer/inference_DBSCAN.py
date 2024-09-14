import os
import numpy as np
import matplotlib.pyplot as plt
import glob

import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from vae import VAE
from utils import *
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial import ConvexHull
import emcee
from sklearn.metrics import mutual_info_score

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--output', type=str, default='results', help='File path of output images')
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion-mnist'], help='The name of dataset')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    parser.add_argument('--add_noise', type=bool, default=False, help='boolean for adding noise to input image')
    parser.add_argument('--noise_factor', type=float, default=0.7, help='Factor of noise')
    parser.add_argument('--dim_z', type=int, default=2, help='Dimension of latent vector')  # , required=True)
    parser.add_argument('--n_hidden', type=int, default=500, help='Number of hidden units in MLP')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate of adam optimizer')
    parser.add_argument('--num_epochs', type=int, default=10, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

    parser.add_argument('--PRR_n_img_x', type=int, default=10, help='Number of images along x-axis')
    parser.add_argument('--PRR_n_img_y', type=int, default=10, help='Number of images along y-axis')
    parser.add_argument('--PMLR_n_img_x', type=int, default=20, help='Number of images along x-axis')
    parser.add_argument('--PMLR_n_img_y', type=int, default=20, help='Number of images along y-axis')
    parser.add_argument('--PMLR_z_range', type=float, default=2.0, help='Range for uniformly distributed latent vector')
    parser.add_argument('--PMLR_n_samples', type=int, default=5000,
                        help='Number of samples in order to get distribution of labeled data')
    parser.add_argument('--model_dir', type=str, default='model', help='Directory to save the trained model')

    return check_args(parser.parse_args())


def check_args(args):
    # --output
    try:
        os.mkdir(args.output)
    except(FileExistsError):
        pass
    # delete all output files
    # files = glob.glob(args.output + '/*')
    # for file in files:
    #     os.remove(file)

    # --model_dir
    try:
        os.mkdir(args.model_dir)
    except(FileExistsError):
        pass

    # --add_noise
    try:
        assert args.add_noise == True or args.add_noise == False
    except:
        print('add_noise must be boolean type')
        return None

    # --dim-z
    try:
        assert args.dim_z > 0
    except:
        print('dim_z must be positive interger')

    # --n_hidden
    try:
        assert args.n_hidden >= 1
    except:
        print('number of hidden units must be larger than or equal to one')

    # --learning_rate
    try:
        assert args.learning_rate > 0
    except:
        print('learning_rate must be positive')

    # --num_epochs
    try:
        assert args.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')

    return args

# Assuming the VAE class and related functions are already defined as per your code

def plot_latent_space(vae, n=30, figsize=15):
    """Plots a 2D manifold of the digits"""
    digit_size = 28
    scale = 2.0
    figure = np.zeros((digit_size * n, digit_size * n))

    # Linearly spaced coordinates on the unit square were transformed
    # through the inverse CDF (ppf) of the Gaussian to produce values
    # of the latent variables z, since the prior of the latent space
    # is Gaussian
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(figsize, figsize))
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


def plot_reconstructed_images(vae, x_test, n=10):
    """Plots original and reconstructed images"""
    x_test = x_test[:n]
    x_reconstructed, _, _ = vae(x_test)

    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(x_reconstructed[i].numpy().reshape(28, 28), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()


def log_prob_fn(z, data, step_size):
    """Log-probability function for MCMC sampling.

    Args:
        z (np.ndarray): The current position in latent space.
        data (np.ndarray): The data points in latent space.
        step_size (float): Step size for the Gaussian kernel.

    Returns:
        float: The log-probability.
    """
    distances = np.linalg.norm(data - z, axis=1)
    density = np.sum(np.exp(-distances ** 2 / (2 * step_size ** 2)))
    return np.log(density + 1e-10)  # Add a small value to avoid log(0)


def run_mcmc(data, n_samples=100, step_size=0.1):
    """Run MCMC using emcee to sample points.

    Args:
        data (np.ndarray): The data points in latent space.
        n_samples (int): Number of points to sample.
        step_size (float): Step size for the proposal distribution.

    Returns:
        np.ndarray: Array of sampled points.
    """
    n_walkers = 10  # Number of walkers (parallel MCMC chains)
    n_dim = data.shape[1]  # Dimensionality of the latent space

    # Initial positions of the walkers (randomly selected data points)
    starting_guesses = data[np.random.choice(len(data), n_walkers)]

    # Create the sampler object
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob_fn, args=(data, step_size))

    # Run the MCMC chain
    sampler.run_mcmc(starting_guesses, n_samples)

    # Flatten the chain (concatenate all samples from all walkers)
    samples = sampler.get_chain(flat=True)

    return samples


def compute_mutual_information(img1, img2):
    """Compute mutual information between two images.

    Args:
        img1 (np.ndarray): First image.
        img2 (np.ndarray): Second image.

    Returns:
        float: The mutual information score.
    """
    # Flatten the images into 1D arrays
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()

    # Compute mutual information
    mi = mutual_info_score(img1_flat, img2_flat)

    return mi

def main(args):
    # Load the MNIST dataset
    x_train, _, x_test, y_test, input_shape = prepare_dataset(args)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype('float32') / 255.
    x_test = np.reshape(x_test, (len(x_test), 28 * 28))

    # Instantiate the VAE model with the same args and input_shape used during training
    vae = VAE(input_shape, args=args)

    # Load the trained weights
    encoder_save_path = os.path.join(args.model_dir, 'vae_encoder_weights.h5')
    decoder_save_path = os.path.join(args.model_dir, 'vae_decoder_weights.h5')
    vae.load_model(encoder_save_path, decoder_save_path)

    # # Plot the latent space
    # plot_latent_space(vae)
    #
    # # Plot reconstructed images
    # plot_reconstructed_images(vae, x_test)

    # Load the provided CSV file
    file_path = r'E:\Yuli\Projects\FM\FM_selection\Vote_MI-MNIST-main\results\PMLR_map_epoch_75.csv'
    data = pd.read_csv(file_path)

    # Extract the relevant columns for clustering
    X = data[['z1', 'z2']]

    # Perform DBSCAN clustering and # Count the number of clusters found
    dbscan = DBSCAN(eps=0.18, min_samples=25)
    data['cluster'] = dbscan.fit_predict(X)
    num_clusters = len(set(data['cluster'])) - (1 if -1 in data['cluster'] else 0)

    # Create subplots for before and after clustering
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # Dot plot before clustering
    # Draw contours around each cluster
    for cluster in range(10):
        cluster_points = X[data['category'] == cluster]
        if len(cluster_points) > 2:  # ConvexHull requires at least 3 points
            hull = ConvexHull(cluster_points)
            for simplex in hull.simplices:
                axes[0].plot(cluster_points.iloc[simplex, 0], cluster_points.iloc[simplex, 1], 'k-', lw=2)
    scatter1 = axes[0].scatter(X['z1'], X['z2'], c=data['category'], cmap='viridis', marker='o', alpha=0.6, edgecolor='w', s=10)
    axes[0].set_title('Dot Plot of Latent Space before Clusters')
    axes[0].set_xlabel('z1')
    axes[0].set_ylabel('z2')
    fig.colorbar(scatter1, ax=axes[0], label='Original Categories')

    # Dot plot after clustering
    scatter2 = axes[1].scatter(X['z1'], X['z2'], c=data['cluster'], cmap='viridis', marker='o', alpha=0.6,
                               edgecolor='w', s=10)

    # Draw contours around each cluster
    for cluster in range(10):
        cluster_points = X[data['cluster'] == cluster]
        if len(cluster_points) > 2:  # ConvexHull requires at least 3 points
            hull = ConvexHull(cluster_points)
            for simplex in hull.simplices:
                axes[1].plot(cluster_points.iloc[simplex, 0], cluster_points.iloc[simplex, 1], 'k-', lw=2)

    axes[1].set_title('Dot Plot of Latent Space with Clusters')
    axes[1].set_xlabel('z1')
    axes[1].set_ylabel('z2')
    fig.colorbar(scatter2, ax=axes[1], label='Cluster Label')

    # Show the combined plot
    plt.show()

    # Randomly sample 100 points from each category
    sampled_data_random = data.sample(n=100, random_state=42)

    # MCMC sampling strategy
    sampled_data = []

    for cluster in range(num_clusters):
        cluster_points = data[data['cluster'] == cluster][['z1', 'z2']].values
        if len(cluster_points) < 5:
            continue

        # Run MCMC sampling to get 5 initial points
        mcmc_samples = run_mcmc(cluster_points, n_samples=5)

        # Compute mutual information and find the rest of the points
        cluster_images = vae.decoder.predict(cluster_points)
        mcmc_images = vae.decoder.predict(mcmc_samples)
        mutual_infos = []

        for i, img in enumerate(cluster_images):
            mi = min(compute_mutual_information(img, mcmc_img) for mcmc_img in mcmc_images)
            mutual_infos.append((mi, cluster_points[i]))

        # Sort by mutual information and select points with the highest MI
        mutual_infos.sort(key=lambda x: -x[0])

        # Select additional points: 10% of the cluster size
        num_additional_samples = max(0, int(len(cluster_points) * 0.10) - 5)
        selected_points = [mi[1] for mi in mutual_infos[:num_additional_samples]]
        sampled_data.extend(mcmc_samples.tolist() + selected_points)

    # Now perform MCMC sampling on the extended sampled data as a new "category"
    sampled_data = np.array(sampled_data)
    final_samples = run_mcmc(sampled_data, n_samples=10)

    # Compute mutual information for the 10 sampled points with all other points
    final_images = vae.decoder.predict(final_samples)
    all_images = vae.decoder.predict(sampled_data)
    final_mutual_infos = []

    for i, img in enumerate(all_images):
        mi = max(compute_mutual_information(img, final_img) for final_img in final_images)
        final_mutual_infos.append((mi, sampled_data[i]))

    # Sort by mutual information and select the 90 points with the lowest MI
    final_mutual_infos.sort(key=lambda x: x[0])  # Sort in ascending order of MI
    num_second_stage_samples = max(1, int(len(sampled_data) * 0.10))
    selected_final_points = [mi[1] for mi in final_mutual_infos[:num_second_stage_samples]]

    # Combine the 10 MCMC samples and the 90 highest MI points
    final_selected_data = np.vstack([final_samples, selected_final_points])

    # Convert the final selected data to DataFrame for further processing
    final_selected_df = pd.DataFrame(final_selected_data, columns=['z1', 'z2'])
    final_selected_df['category'] = data['category'].iloc[:len(final_selected_df)]
    final_selected_df['cluster'] = data['cluster'].iloc[:len(final_selected_df)]

    # Save sampled data to CSV
    #final_selected_df.to_csv('results\PMLR_map_epoch_75_final_sampled_points.csv', index=False)

    # Create subplots for sampled points before and after clustering
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    # Dot plot of sampled points before clustering
    axes[0].scatter(sampled_data_random['z1'], sampled_data_random['z2'], c=sampled_data_random['category'], cmap='viridis', marker='o', alpha=0.6, edgecolor='w', s=30)
    axes[0].set_title('Sampled Points from Latent Space before Clusters')
    axes[0].set_xlabel('z1')
    axes[0].set_ylabel('z2')

    # Dot plot of sampled points after clustering
    axes[1].scatter(final_selected_df['z1'], final_selected_df['z2'], c=final_selected_df['cluster'], cmap='viridis', marker='o', alpha=0.6, edgecolor='w', s=30)
    axes[1].set_title('Sampled Points from Latent Space with Clusters')
    axes[1].set_xlabel('z1')
    axes[1].set_ylabel('z2')

    # Show the combined plot of sampled points
    plt.show()

    # Ensure the output directory exists
    selected_image_folder = "images\selected_image_folder"
    all_image_folder = 'images\image_folder'
    os.makedirs(selected_image_folder, exist_ok=True)
    os.makedirs(all_image_folder, exist_ok=True)

    # Add a column to identify the row number in final_selected_df
    final_selected_df['No'] = range(1, len(final_selected_df) + 1)
    data['No'] = range(1, len(data) + 1)

    # Send the selected sampled latent vectors back to the decoder for reconstruction
    sampled_latent_vectors = final_selected_df[['z1', 'z2']].values
    reconstructed_images = vae.decoder.predict(sampled_latent_vectors)

    # Send the all sampled latent vectors back to the decoder for reconstruction
    all_sampled_latent_vectors = data[['z1', 'z2']].values
    all_reconstructed_images = vae.decoder.predict(all_sampled_latent_vectors)


    # Save each selected reconstructed image to the specified folder with the row number as the filename
    for i, img in enumerate(reconstructed_images):
        img_reshaped = img.reshape(28, 28)  # Reshape to original image dimensions
        img_no = final_selected_df.iloc[i]['No']  # Get the corresponding row number
        img_filename = os.path.join(selected_image_folder, f"image_{int(img_no)}.png")

        # Save the image using matplotlib's imsave
        plt.imsave(img_filename, img_reshaped, cmap='gray')

    # Save all reconstructed image to the specified folder with the row number as the filename
    for i, img in enumerate(all_reconstructed_images):
        img_reshaped = img.reshape(28, 28)  # Reshape to original image dimensions
        img_no = data.iloc[i]['No']  # Get the corresponding row number
        img_filename = os.path.join(all_image_folder, f"image_{int(img_no)}.png")

        # Save the image using matplotlib's imsave
        plt.imsave(img_filename, img_reshaped, cmap='gray')

    # Optional: Plot the reconstructed images, handling cases where there are more than 100 images
    n = 10  # Number of images to display per row
    total_images = len(reconstructed_images)

    # Calculate the number of figures needed
    figs_needed = (total_images // 100) + 1 if total_images % 100 != 0 else total_images // 100

    for fig_num in range(figs_needed):
        plt.figure(figsize=(20, 20))

        start_idx = fig_num * 100
        end_idx = min(start_idx + 100, total_images)

        for i in range(start_idx, end_idx):
            ax = plt.subplot(10, n, (i % 100) + 1)
            plt.imshow(reconstructed_images[i].reshape(28, 28), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.suptitle(f'Reconstructed Images from Final Selected Samples (Figure {fig_num + 1})', fontsize=16)
        plt.show()

    # Optionally, save the final_selected_df to a CSV
    final_selected_df.to_csv('images\PMLR_map_epoch_75_final_sampled_points_with_NO.csv',
                             index=False)



if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    if args is None:
        exit()

    main(args)
