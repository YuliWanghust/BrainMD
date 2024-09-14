# Variational Auto-Encoder for MNIST

This is an implementation of variational auto-encoder(VAE) for MNIST using Vote-MI

## Usage
Recommend to use Python==3.8.18 or later version

```
conda create --name <env>
pip install -r requirements.txt
```

### The requirements should at least include...
```
tensorflow >= 2.0
numpy
argparse
matplotlib
cv2
```

### Command
```
python main.py --[argument] <value>
```

*Example* : `python main.py --dim_z 4`

### Arguments

* `--dim_z` : Dimension of latent vector. *Default* : `4`
* `--output` : File path of output images. *Default* : `results`
* `--dataset` : The name of Dataset. List : ['mnist', 'fashion-mnist']. *Default* : `mnist`
* `--seed` : Random Seed. *Default* : `0`
* `--add_noise` : Boolean for adding noise to input image(gaussian noise). *Default* : `False`
* `--noise_factor` : Factor of gaussian noise. *Default* : `0.7`
* `--n_hidden` : The number of hidden units in MLP. *Default* : `500`
* `--learning_rate` : Learning rate of adam optimizer. *Default* : `1e-3`
* `--num_epochs` : The number of epochs to run. *Default* : `40`
* `--batch_size` : Batch size. *Default* : `64`

* `--PMLR_z_range` : Rnage for uniformly distributed latent. *Default* : `2.0`
* `--PMLR_n_samples` : Number of samples in order to get distribution of labeled data. *Default* : `5000`
* `--model_dir` : Directory to save the trained model. *Default* : `model`

## Results

### VAE Training

Well trained VAE must be able to reproduce input images.

Images below show the performance of learned generative models for dimensionality dim_z=2.

Command : `python main.py --dim_z 2 --num_epochs 100 --model_dir model`

|Input image|Reconstructed image from 2-D latent space|
|:---:|:---:|
|<img src="github_images/input.jpg">|<img src="github_images/dim_z_2.jpg">|

|2-D latent space (epoch 1)|2-D latent space (epoch 40)|2-D latent space (epoch 100)|
|:---:|:---:|:---:|
|<img src="github_images/PMLR_epoch_1.PNG">|<img src="github_images/PMLR_epoch_40.PNG">|<img src="github_images/PMLR_epoch_100.PNG">|

### After finishing the VAE training

You will have the following files under the folder of `results` and `model`:

Under `results` folder:

* `PMLR_map_epoch_XX.pdf` : Latent space plotting for each epoch.
* `PMLR_map_epoch_XX.csv` : MNIST testing set projected points in latent space for each epoch
* `PRR_epoch_XX.jpg`: Example of reconstructed images from trained VAE for epoch

Under `model` folder:
* `vae_decoder_weights.h5`: trained VAE weights of the decoder part
* `vae_encoder_weights.h5`: trained VAE weights of the encoder part


### Implement Vote-MI

After the VAE model is trained, we then implement Vote-MI on the latent space of VAE

Command : `python inference.py --dim_z 2 --num_epochs 10 --model_dir model`

You can see two results, when Vote-MI implemented:

|Original latent space v.s. Clustered Latent space using DBSCAN|
|:---:|
|<img src="github_images/Clusters.PNG">|

|Selected points from random samples v.s. Selected points from Vote-MI|
|:---:|
|<img src="github_images/selected.png">|

### After finishing the Vote-MI selection

You will have the following files under the folder of `images`:

Under `images` folder:

* `image_folder` : all images before the selection.
* `selected_image_folder` : representative images from Vote-MI
* `.excel`: latent space information, category and GT label for the selected images

## Downstream tasks

The selected images can be used for a variety of downstream tasks then.
