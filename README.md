***News***

- 8 April 2019: We re-implement these GANs by **Tensorflow 2**! The old version is here: [v1](https://github.com/LynnHo/DCGAN-LSGAN-WGAN-WGAN-GP-Tensorflow/tree/v1) or in the "v1" directory.
- [**PyTorch** Version](https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch)

<hr style="height:1px" />

<p align="center">
    <img src="./pics/celeba_dragan.gif" width="49.7%" />  <img src="./pics/anime_dragan.gif" width="49.7%" />
</p>

<hr style="height:1px" />

# <p align="center"> GANs - Tensorflow 2 </p>

Tensorflow 2 implementations of [DCGAN](https://arxiv.org/abs/1511.06434), [LSGAN](https://arxiv.org/abs/1611.04076), [WGAN-GP](http://arxiv.org/abs/1704.00028) and [DRAGAN](https://arxiv.org/abs/1705.07215v5).

## Exemplar results

### Fashion-MNIST

DCGAN                                    | LSGAN                                      | WGAN-GP                                      | DRAGAN
:---:                                    | :---:                                      | :---:                                        | :---:
<img src="./pics/fashion-mnist_gan.jpg"> | <img src="./pics/fashion-mnist_lsgan.jpg"> | <img src="./pics/fashion-mnist_wgan-gp.jpg"> | <img src="./pics/fashion-mnist_dragan.jpg">

### CelebA

DCGAN                                 | LSGAN
:---:                                 | :---:
<img src="./pics/celeba_gan.jpg">     | <img src="./pics/celeba_lsgan.jpg">
**WGAN-GP**                           | **DRAGAN**
<img src="./pics/celeba_wgan-gp.jpg"> | <img src="./pics/celeba_dragan.jpg">

### Anime

**WGAN-GP**                          | **DRAGAN**
:---:                                | :---:
<img src="./pics/anime_wgan-gp.jpg"> | <img src="./pics/anime_dragan.jpg">

# Usage

- Environment

    - Python 3.6

    - TensorFlow 2.2, TensorFlow Addons 0.10.0

    - OpenCV, scikit-image, tqdm, oyaml

    - *we recommend [Anaconda](https://www.anaconda.com/distribution/#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers), then you can create the TensorFlow 2.2 environment with commands below*

        ```console
        conda create -n tensorflow-2.2 python=3.6

        source activate tensorflow-2.2

        conda install scikit-image tqdm tensorflow-gpu=2.2

        conda install -c conda-forge oyaml

        pip install tensorflow-addons==0.10.0
        ```

    - *NOTICE: if you create a new conda environment, remember to activate it before any other command*

        ```console
        source activate tensorflow-2.2
        ```

- Datasets

    - Fashion-MNIST will be automatically downloaded
    - CelebA should be prepared by yourself in **./data/img_align_celeba/\*.jpg**
        - dataset link (find "img_align_celeba.zip"):
            - [Baidu Netdisk](https://pan.baidu.com/s/1CRxxhoQ97A5qbsKO7iaAJg) (password rp0s) or
            - [Google Drive](https://drive.google.com/drive/folders/0B7EVK8r0v71pTUZsaXdaSnZBZzg)
    - the Anime dataset should be prepared by yourself in **./data/faces/\*.jpg**
        - dataset link: https://www.kaggle.com/splcher/animefacedataset

- Examples of training

    - Fashion-MNIST DCGAN

        ```console
        CUDA_VISIBLE_DEVICES=0 python train.py --dataset=fashion_mnist --epoch=25 --adversarial_loss_mode=gan
        ```

    - CelebA DRAGAN

        ```console
        CUDA_VISIBLE_DEVICES=0 python train.py --dataset=celeba --epoch=25 --adversarial_loss_mode=gan --gradient_penalty_mode=dragan
        ```

    - Anime WGAN-GP

        ```console
        CUDA_VISIBLE_DEVICES=0 python train.py --dataset=anime --epoch=200 --adversarial_loss_mode=wgan --gradient_penalty_mode=wgan-gp --n_d=5
        ```

    - see more training exampls in [commands.sh](./commands.sh)

    - tensorboard for loss visualization

        ```console
        tensorboard --logdir ./output/fashion_mnist_gan/summaries --port 6006
        ```

