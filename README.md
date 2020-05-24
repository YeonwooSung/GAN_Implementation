# GAN_Implementation

## Overview

Basically, this repository is a collection of my PyTorch implementation of Generative Adversarial Network varieties presented in research papers.

## Table of Contents

  * [Installation](#Installation)
  * [Implementations](#Implementations)
    + [CycleGAN](#cyclegan)
    + [GAN](#gan)
    + [Wasserstein GAN](#wasserstein-gan)
    + [Wasserstein GAN GP](#wasserstein-gan-gp)

## Installation

    $ git clone https://github.com/eriklindernoren/PyTorch-GAN
    $ cd PyTorch-GAN/
    $ sudo pip3 install -r requirements.txt

## Implementations

### CycleGAN

_Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks_

#### Authors

Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros

#### Abstract

Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G: X -> Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F: Y -> X and introduce a cycle consistency loss to push F(G(X)) ~ X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach.

[[paper]](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html) [[code]](./src/cyclegan/cyclegan.py)

#### Example Running

```
$ cd src/cyclegan/
$ bash download_dataset.sh monet2photo
$ python3 cyclegan.py
```

Please make sure that you download the suitable dataset before you run the python script for the cyclegan.

### GAN

_Generative Adversarial Network_

#### Authors

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio

#### Abstract

We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.

[[paper]](https://arxiv.org/abs/1406.2661)  [[code]](./src/gan/gan.py)

#### Example Running

```
$ cd src/gan/
$ python3 gan.py
```

### Wasserstein GAN

_Wasserstein GAN_

#### Authors

Martin Arjovsky, Soumith Chintala, Léon Bottou

#### Abstract

We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to other distances between distributions.

[[Paper]](https://arxiv.org/abs/1701.07875) [[Code]](./src/wgan/wgan.py)

#### Example Running

```
$ cd src/wgan/
$ python3 wgan.py
```

### Wasserstein GAN GP

_Improved training of Wasserstein GANs_

#### Authors

Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron C. Courville

#### Abstract

Generative Adversarial Networks (GANs) are powerful generative models, but suffer from training instability. The recently proposed Wasserstein GAN (WGAN) makes progress toward stable training of GANs, but sometimes can still generate only poor samples or fail to converge. We find that these problems are often due to the use of weight clipping in WGAN to enforce a Lipschitz constraint on the critic, which can lead to undesired behavior. We propose an alternative to clipping weights: penalize the norm of gradient of the critic with respect to its input. Our proposed method performs better than standard WGAN and enables stable training of a wide variety of GAN architectures with almost no hyperparameter tuning, including 101-layer ResNets and language models with continuous generators. We also achieve high quality generations on CIFAR-10 and LSUN bedrooms.

[[Paper]](http://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans) [[Code]](./src/wgan_gp/wgan_gp.py)

#### Example Running

```
$ cd src/wgan_gp/
$ python3 wgan_gp.py
```
