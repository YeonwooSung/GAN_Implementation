# GAN_Implementation

## Overview

Basically, this repository is a collection of my PyTorch implementation of Generative Adversarial Network varieties presented in research papers.

If you liked my repository, please star it. Also, contributions are always welcomed, so please give in touch.

## ToDo

1. DualStyleGAN
2. TediGAN

## Table of Contents

  * [Installation](#Installation)
  * [Implementations](#Implementations)
    + [BEGAN](#began)
    + [Coupled GAN](#coupled-gan)
    + [CycleGAN](#cyclegan)
    + [DCGAN](#dcgan)
    + [DRAGAN](#dragan)
    + [Energy-Based GAN](#energy-based-gan)
    + [GAN](#gan)
    + [MUNIT](#munit)
    + [PGGAN](#pggan)
    + [SAGAN](#sagan)
    + [Semi-Supervised GAN](#semi-supervised-gan)
    + [Softmax GAN](#softmax-gan)
    + [StarGAN](#stargan)
    + [StoryGAN](#storygan)
    + [TransGAN](#transgan)
    + [TUNIT](#tunit)
    + [UNIT](#unit)
    + [Wasserstein GAN](#wasserstein-gan)
    + [Wasserstein GAN GP](#wasserstein-gan-gp)
    + [WaveGAN](#wavegan)

## Installation

    $ git clone https://github.com/YeonwooSung/GAN_Implementation
    $ cd PyTorch-GAN/
    $ sudo pip3 install -r requirements.txt

The WaveGAN uses libav, thus, you should install the libav before training the WaveGAN.

## Implementations

### BEGAN

_BEGAN: Boundary Equilibrium Generative Adversarial Networks_

#### Authors (BEGAN)

David Berthelot, Thomas Schumm, Luke Metz

#### Abstract (BEGAN)

We propose a new equilibrium enforcing method paired with a loss derived from the Wasserstein distance for training auto-encoder based Generative Adversarial Networks. This method balances the generator and discriminator during training. Additionally, it provides a new approximate convergence measure, fast and stable training and high visual quality. We also derive a way of controlling the trade-off between image diversity and visual quality. We focus on the image generation task, setting a new milestone in visual quality, even at higher resolutions. This is achieved while using a relatively simple model architecture and a standard training procedure.

[[Paper]](https://arxiv.org/abs/1703.10717) [[Code]](./src/began.py)

#### Running Example

```
$ cd src/began/
$ python3 began.py
```

### Coupled GAN

_Coupled Generative Adversarial Networks_

#### Authors (CoGAN)

Ming-Yu Liu, Oncel Tuzel

#### Abstract (CoGAN)

We propose coupled generative adversarial network (CoGAN) for learning a joint distribution of multi-domain images. In contrast to the existing approaches, which require tuples of corresponding images in different domains in the training set, CoGAN can learn a joint distribution without any tuple of corresponding images. It can learn a joint distribution with just samples drawn from the marginal distributions. This is achieved by enforcing a weight-sharing constraint that limits the network capacity and favors a joint distribution solution over a product of marginal distributions one. We apply CoGAN to several joint distribution learning tasks, including learning a joint distribution of color and depth images, and learning a joint distribution of face images with different attributes. For each task it successfully learns the joint distribution without any tuple of corresponding images. We also demonstrate its applications to domain adaptation and image transformation.

[[paper]](https://arxiv.org/abs/1606.07536) [[code]](./src/cogan/cogan)

#### Example Running (CoGAN)

```
$ cd src/cogan/
$ python3 cogan.py
```

### CycleGAN

_Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks_

#### Authors (CycleGAN)

Jun-Yan Zhu, Taesung Park, Phillip Isola, Alexei A. Efros

#### Abstract (CycleGAN)

Image-to-image translation is a class of vision and graphics problems where the goal is to learn the mapping between an input image and an output image using a training set of aligned image pairs. However, for many tasks, paired training data will not be available. We present an approach for learning to translate an image from a source domain X to a target domain Y in the absence of paired examples. Our goal is to learn a mapping G: X -> Y such that the distribution of images from G(X) is indistinguishable from the distribution Y using an adversarial loss. Because this mapping is highly under-constrained, we couple it with an inverse mapping F: Y -> X and introduce a cycle consistency loss to push F(G(X)) ~ X (and vice versa). Qualitative results are presented on several tasks where paired training data does not exist, including collection style transfer, object transfiguration, season transfer, photo enhancement, etc. Quantitative comparisons against several prior methods demonstrate the superiority of our approach.

[[paper]](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.html) [[code]](./src/cyclegan/cyclegan.py)

#### Example Running (CycleGAN)

```
$ cd src/cyclegan/
$ bash download_dataset.sh monet2photo
$ python3 cyclegan.py
```

Please make sure that you download the suitable dataset before you run the python script for the cyclegan.

### DCGAN

_Deep Convolutional Generative Adversarial Network_

#### Authors (DCGAN)

Alec Radford, Luke Metz, Soumith Chintala

#### Abstract (DCGAN)

In recent years, supervised learning with convolutional networks (CNNs) has seen huge adoption in computer vision applications. Comparatively, unsupervised learning with CNNs has received less attention. In this work we hope to help bridge the gap between the success of CNNs for supervised learning and unsupervised learning. We introduce a class of CNNs called deep convolutional generative adversarial networks (DCGANs), that have certain architectural constraints, and demonstrate that they are a strong candidate for unsupervised learning. Training on various image datasets, we show convincing evidence that our deep convolutional adversarial pair learns a hierarchy of representations from object parts to scenes in both the generator and discriminator. Additionally, we use the learned features for novel tasks - demonstrating their applicability as general image representations.

[[Paper]](https://arxiv.org/abs/1511.06434) [[Code]](./src/dcgan/dcgan.py)

#### Example Running (DCGAN)

```
$ cd src/dcgan/
$ python3 dcgan.py
```

### DRAGAN

_On Convergence and Stability of GANs_

#### Authors (DRAGAN)

Naveen Kodali, Jacob Abernethy, James Hays, Zsolt Kira

#### Abstract (DRAGAN)

We propose studying GAN training dynamics as regret minimization, which is in contrast to the popular view that there is consistent minimization of a divergence between real and generated distributions. We analyze the convergence of GAN training from this new point of view to understand why mode collapse happens. We hypothesize the existence of undesirable local equilibria in this non-convex game to be responsible for mode collapse. We observe that these local equilibria often exhibit sharp gradients of the discriminator function around some real data points. We demonstrate that these degenerate local equilibria can be avoided with a gradient penalty scheme called DRAGAN. We show that DRAGAN enables faster training, achieves improved stability with fewer mode collapses, and leads to generator networks with better modeling performance across a variety of architectures and objective functions.

[[Paper]](https://arxiv.org/abs/1705.07215) [[Code]](./src/dragan/dragan.py)

#### Example Running

```
$ cd src/dragan
$ python3 dragan.py
```

### Energy-Based GAN

_Energy-based Generative Adversarial Network_

#### Authors (Energy-Based GAN)

Junbo Zhao, Michael Mathieu, Yann LeCun

#### Abstract (Energy-Based GAN)

We introduce the "Energy-based Generative Adversarial Network" model (EBGAN) which views the discriminator as an energy function that attributes low energies to the regions near the data manifold and higher energies to other regions. Similar to the probabilistic GANs, a generator is seen as being trained to produce contrastive samples with minimal energies, while the discriminator is trained to assign high energies to these generated samples. Viewing the discriminator as an energy function allows to use a wide variety of architectures and loss functionals in addition to the usual binary classifier with logistic output. Among them, we show one instantiation of EBGAN framework as using an auto-encoder architecture, with the energy being the reconstruction error, in place of the discriminator. We show that this form of EBGAN exhibits more stable behavior than regular GANs during training. We also show that a single-scale architecture can be trained to generate high-resolution images.

[[paper]](https://arxiv.org/abs/1609.03126) [[code]](./src/ebgan/ebgan.py)

#### Example Running (Energy-Based GAN)

```
$ cd src/ebgan/
$ python3 ebgan.py
```

### GAN

_Generative Adversarial Network_

#### Authors (GAN)

Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio

#### Abstract (GAN)

We propose a new framework for estimating generative models via an adversarial process, in which we simultaneously train two models: a generative model G that captures the data distribution, and a discriminative model D that estimates the probability that a sample came from the training data rather than G. The training procedure for G is to maximize the probability of D making a mistake. This framework corresponds to a minimax two-player game. In the space of arbitrary functions G and D, a unique solution exists, with G recovering the training data distribution and D equal to 1/2 everywhere. In the case where G and D are defined by multilayer perceptrons, the entire system can be trained with backpropagation. There is no need for any Markov chains or unrolled approximate inference networks during either training or generation of samples. Experiments demonstrate the potential of the framework through qualitative and quantitative evaluation of the generated samples.

[[paper]](https://arxiv.org/abs/1406.2661)  [[code]](./src/gan/gan.py)

#### Example Running (GAN)

```
$ cd src/gan/
$ python3 gan.py
```

### MUNIT

_Multimodal Unsupervised Image-to-Image Translation_

#### Authors (MUNIT)

Xun Huang, Ming-Yu Liu, Serge Belongie, Jan Kautz

#### Abstract (MUNIT)

Unsupervised image-to-image translation is an important and challenging problem in computer vision. Given an image in the source domain, the goal is to learn the conditional distribution of corresponding images in the target domain, without seeing any pairs of corresponding images. While this conditional distribution is inherently multimodal, existing approaches make an overly simplified assumption, modeling it as a deterministic one-to-one mapping. As a result, they fail to generate diverse outputs from a given source domain image. To address this limitation, we propose a Multimodal Unsupervised Image-to-image Translation (MUNIT) framework. We assume that the image representation can be decomposed into a content code that is domain-invariant, and a style code that captures domain-specific properties. To translate an image to another domain, we recombine its content code with a random style code sampled from the style space of the target domain. We analyze the proposed framework and establish several theoretical results. Extensive experiments with comparisons to the state-of-the-art approaches further demonstrates the advantage of the proposed framework. Moreover, our framework allows users to control the style of translation outputs by providing an example style image. Code and pretrained models are available at [this https URL](https://github.com/nvlabs/MUNIT).

[[Paper]](https://arxiv.org/abs/1804.04732) [[Code]](./src/munit/munit.py)

#### Example Running (MUNIT)

```
$ cd data/
$ bash download_pix2pix_dataset.sh edges2shoes
$ cd ../src/munit/
$ python3 munit.py --dataset_name edges2shoes
```

### PGGAN

_Progressive Growing of GANs for Improved Quality, Stability, and Variation_

#### Authors (PGGAN)

Tero Karras, Timo Aila, Samuli Laine, Jaakko Lehtinen

#### Abstract (PGGAN)

We describe a new training methodology for generative adversarial networks. The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality, e.g., CelebA images at 1024². We also propose a simple way to increase the variation in generated images, and achieve a record inception score of 8.80 in unsupervised CIFAR10. Additionally, we describe several implementation details that are important for discouraging unhealthy competition between the generator and discriminator. Finally, we suggest a new metric for evaluating GAN results, both in terms of image quality and variation. As an additional contribution, we construct a higher-quality version of the CelebA dataset.

[[paper]](https://research.nvidia.com/publication/2017-10_Progressive-Growing-of) [[Code]](./src/pggan/main.py)

#### Example Running (PGGAN)

Before running the "main.py", you need to download the dataset from [here](https://drive.google.com/drive/folders/1j6uZ_a6zci0HyKZdpDq9kSa8VihtEPCp) to '/data' directory. You could find more information about downloading dataset from [the official PGGAN repository](https://github.com/tkarras/progressive_growing_of_gans/). My implementation uses the celeb dataset, so if you want to use other dataset, please follow the instructions in [the official PGGAN repository](https://github.com/tkarras/progressive_growing_of_gans/).

```
$ cd src/pggan
$ python3 main.py
```

### SAGAN

_Self-Attention Generative Adversarial Networks_

#### Authors (SAGAN)

Han Zhang, Ian Goodfellow, Dimitris Metaxas, Augustus Odena

#### Abstract (SAGAN)

In this paper, we propose the Self-Attention Generative Adversarial Network (SAGAN) which allows attention-driven, long-range dependency modeling for image generation tasks. Traditional convolutional GANs generate high-resolution details as a function of only spatially local points in lower-resolution feature maps. In SAGAN, details can be generated using cues from all feature locations. Moreover, the discriminator can check that highly detailed features in distant portions of the image are consistent with each other. Furthermore, recent work has shown that generator conditioning affects GAN performance. Leveraging this insight, we apply spectral normalization to the GAN generator and find that this improves training dynamics. The proposed SAGAN achieves the state-of-the-art results, boosting the best published Inception score from 36.8 to 52.52 and reducing Frechet Inception distance from 27.62 to 18.65 on the challenging ImageNet dataset. Visualization of the attention layers shows that the generator leverages neighborhoods that correspond to object shapes rather than local regions of fixed shape.

#### Example Running (SAGAN)

Before running the "main.py", you need to download the dataset from [here](https://drive.google.com/drive/folders/1j6uZ_a6zci0HyKZdpDq9kSa8VihtEPCp) to '/data' directory.  I really wanted to find suitable shell script that could help programers to download the CelebA dataset easily. Unfortunately, however, I was not able to find suitable script.

After downloading the dataset in the "data/" directory, now you could run the SAGAN program by following the instruction.

```
$ cd src/sagan
$ python3 main.py
```

#### Contributions (SAGAN)

Codes for the SAGAN in this repository are based on [this repository](https://github.com/heykeetae/Self-Attention-GAN). Credits to [heykeetae](https://github.com/heykeetae).

### Semi-Supervised GAN

_Semi-Supervised Generative Adversarial Network_

#### Authors (Semi-Supervised GAN)

Augustus Odena

#### Abstract (Semi-Supervised GAN)

We extend Generative Adversarial Networks (GANs) to the semi-supervised context by forcing the discriminator network to output class labels. We train a generative model G and a discriminator D on a dataset with inputs belonging to one of N classes. At training time, D is made to predict which of N+1 classes the input belongs to, where an extra class is added to correspond to the outputs of G. We show that this method can be used to create a more data-efficient classifier and that it allows for generating higher quality samples than a regular GAN.

[[Paper]](https://arxiv.org/abs/1606.01583) [[Code]](./src/sgan.py)

#### Running Example (Semi-Supervised GAN)

```
$ cd src/sgan/
$ python3 sgan.py
```

### Softmax GAN

_Softmax GAN_

#### Authors (Softmax GAN)

Min Lin

#### Abstract (Softmax GAN)

Softmax GAN is a novel variant of Generative Adversarial Network (GAN). The key idea of Softmax GAN is to replace the classification loss in the original GAN with a softmax cross-entropy loss in the sample space of one single batch. In the adversarial learning of N real training samples and M generated samples, the target of discriminator training is to distribute all the probability mass to the real samples, each with probability 1M, and distribute zero probability to generated data. In the generator training phase, the target is to assign equal probability to all data points in the batch, each with probability 1M+N. While the original GAN is closely related to Noise Contrastive Estimation (NCE), we show that Softmax GAN is the Importance Sampling version of GAN. We futher demonstrate with experiments that this simple change stabilizes GAN training.

[[Paper](https://arxiv.org/abs/1704.06191)] [[Code](./src/softmax_gan/softmax_gan.py)]

#### Example Running (Softmax GAN)

```
$ cd src/softmax_gan
$ python3 softmax_gan.py
```

### StarGAN

_StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation_

#### Authors (StarGAN)

Yunjey Choi, Minje Choi, Munyoung Kim, Jung-Woo Ha, Sunghun Kim, Jaegul Choo

#### Abstract (StarGAN)

Recent studies have shown remarkable success in image-to-image translation for two domains. However, existing approaches have limited scalability and robustness in handling more than two domains, since different models should be built independently for every pair of image domains. To address this limitation, we propose StarGAN, a novel and scalable approach that can perform image-to-image translations for multiple domains using only a single model. Such a unified model architecture of StarGAN allows simultaneous training of multiple datasets with different domains within a single network. This leads to StarGAN's superior quality of translated images compared to existing models as well as the novel capability of flexibly translating an input image to any desired target domain. We empirically demonstrate the effectiveness of our approach on a facial attribute transfer and a facial expression synthesis tasks.

[[Paper]](https://arxiv.org/abs/1711.09020) [[Code]](./src/stargan/stargan.py)

#### Running Example (StarGAN)

Before running the stargan, you need to download the CelebA dataset from [here](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0). Also, you need to download the [annotation file](https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AAA8YmAHNNU6BEfWMPMfM6r9a/Anno?dl=0&preview=list_attr_celeba.txt). After download both dataset and annotation file, copy 'list_attr_celeba.txt' to folder 'img_align_celeba', and save the folder 'img_align_celeba' to "[data](./data)" directory.

After finish the process above, you could run the StarGan by running the following commands.

```
cd src/stargan
python3 stargan.py
```

### StoryGAN

#### Authors (StoryGAN)

Yitong Li, Zhe Gan, Yelong Shen, Jingjing Liu, Yu Cheng, Yuexin Wu, Lawrence Carin, David Carlson, Jianfeng Gao

#### Abstract (StoryGAN)

We propose a new task, called Story Visualization. Given a multi-sentence paragraph, the story is visualized by generating a sequence of images, one for each sentence. In contrast to video generation, story visualization focuses less on the continuity in generated images (frames), but more on the global consistency across dynamic scenes and characters -- a challenge that has not been addressed by any single-image or video generation methods. We therefore propose a new story-to-image-sequence generation model, StoryGAN, based on the sequential conditional GAN framework. Our model is unique in that it consists of a deep Context Encoder that dynamically tracks the story flow, and two discriminators at the story and image levels, to enhance the image quality and the consistency of the generated sequences. To evaluate the model, we modified existing datasets to create the CLEVR-SV and Pororo-SV datasets. Empirically, StoryGAN outperforms state-of-the-art models in image quality, contextual consistency metrics, and human evaluation.

[[Paper]](https://arxiv.org/abs/1812.02784) [[Code]](./src/storygan/main_clevr.py)

#### Configure File

"/code/cfg/clevr.yml" is the configure file for the model. This file contains the setup of the dimension of the features, maximum training epoches, etc.

#### Running Example (StoryGAN)

```bash
$ cd src/stargan
$ python3 main_clevr.py
```

### TransGAN

_TransGAN: Two Transformers Can Make One Strong GAN_

#### Authors (TransGAN)

Yifan Jiang, Shiyu Chang, Zhangyang Wang

#### Abstract (TransGAN)

The recent explosive interest on transformers has suggested their potential to become powerful "universal" models for computer vision tasks, such as classification, detection, and segmentation. However, how further transformers can go - are they ready to take some more notoriously difficult vision tasks, e.g., generative adversarial networks (GANs)? Driven by that curiosity, we conduct the first pilot study in building a GAN \textbf{completely free of convolutions}, using only pure transformer-based architectures. Our vanilla GAN architecture, dubbed \textbf{TransGAN}, consists of a memory-friendly transformer-based generator that progressively increases feature resolution while decreasing embedding dimension, and a patch-level discriminator that is also transformer-based. We then demonstrate TransGAN to notably benefit from data augmentations (more than standard GANs), a multi-task co-training strategy for the generator, and a locally initialized self-attention that emphasizes the neighborhood smoothness of natural images. Equipped with those findings, TransGAN can effectively scale up with bigger models and high-resolution image datasets. Specifically, our best architecture achieves highly competitive performance compared to current state-of-the-art GANs based on convolutional backbones. Specifically, TransGAN sets \textbf{new state-of-the-art} IS score of 10.10 and FID score of 25.32 on STL-10. It also reaches competitive 8.64 IS score and 11.89 FID score on Cifar-10, and 12.23 FID score on CelebA 64×64, respectively. We also conclude with a discussion of the current limitations and future potential of TransGAN.

[[Paper]](https://arxiv.org/abs/2102.07074) [[Code]](./src/transgan/transgan.py)

### TUNIT

_Rethinking the Truly Unsupervised Image-to-Image Translation_

#### Authors (TUNIT)

Kyungjune Baek, Yunjey Choi, Youngjung Uh, Jaejun Yoo, Hyunjung Shim

#### Abstract (TUNIT)

Every recent image-to-image translation model uses either image-level (i.e. input-output pairs) or set-level (i.e. domain labels) supervision at minimum. However, even the set-level supervision can be a serious bottleneck for data collection in practice. In this paper, we tackle image-to-image translation in a fully unsupervised setting, i.e., neither paired images nor domain labels. To this end, we propose the truly unsupervised image-to-image translation method (TUNIT) that simultaneously learns to separate image domains via an information-theoretic approach and generate corresponding images using the estimated domain labels. Experimental results on various datasets show that the proposed method successfully separates domains and translates images across those domains. In addition, our model outperforms existing set-level supervised methods under a semi-supervised setting, where a subset of domain labels is provided. The source code is available at this [https URL](https://github.com/clovaai/tunit).

[[Paper]](https://arxiv.org/abs/2006.06500) [[Code]](./src/tunit/main.py)

#### Example Running (TUNIT)

Before running the TUNIT, please download the both [AFHQ (StarGANv2)](https://www.dropbox.com/s/t9l9o3vsx2jai3z/afhq.zip?dl=0) and [AnimalFaces (FUNIT)](https://github.com/NVlabs/FUNIT). And put those downloaded dataset into the [data/ directory](./data).

```
# Training
$ python3 main.py --gpu 0 --dataset animal_faces --output_k 10 --data_path '../../data' --p_semi 0.0
$ python3 main.py --gpu 0 --dataset afhq_cat --output_k 10 --data_path '../../data' --p_semi 0.2
$ python3 main.py --gpu 1 --dataset animal_faces --data_path '../../data' --p_semi 1.0
$ python3 main.py --gpu 0,1 --dataset summer2winter --output_k 2 --data_path '../../data' --p_semi 0.0 --img_size 256 --batch_size 16 --ddp


# Test
$ python3 main.py --gpu 0 --dataset animal_faces --output_k 10 --data_path '../../data' --validation --load_model GAN_20190101_101010
$ python3 main.py --gpu 1 --dataset afhq_cat --output_k 10 --data_path '../../data' --validation --load_model GAN_20190101_101010
$ python3 main.py --gpu 2 --dataset summer2winter --output_k 2 --data_path '../../data' --validation --load_model GAN_20190101_101010


# Monitoring - open terminal at ./tunit/logs
$ tensorboard --logdir=./GAN_20200101_101010/events
```

##### Training

Supervised:

```
$ python3 src/tunit/main.py --gpu $GPU_TO_USE --p_semi 1.0 --dataset animal_faces --data_path='../../data'
```

Semi-Supervised:

```
$ python3 main.py --gpu $GPU_TO_USE --p_semi 0.5 --dataset animal_faces --data_path='../../data'
```

Unsupervised:

```
$ python3 main.py --gpu $GPU_TO_USE --p_semi 0.0 --dataset animal_faces --data_path='../../data'
```

##### Testing

```
$ python3 main.py --gpu $GPU_TO_USE --validation --load_model $DIR_TO_LOAD --data_path '../../data' --dataset animal_faces
```

##### Monitoring

```
$ tensorboard --logdir=$DIR/events --port=$PORT
```

#### Contributions (TUNIT)

Most of the codes for the TUNIT model in this repository are from [the TUNIT authors' repository](https://github.com/clovaai/tunit). I simply modified some of the codes to make it much more efficient. Credits to ClovaAI.

### Wasserstein GAN

_Wasserstein GAN_

#### Authors (WGAN)

Martin Arjovsky, Soumith Chintala, Léon Bottou

#### Abstract (WGAN)

We introduce a new algorithm named WGAN, an alternative to traditional GAN training. In this new model, we show that we can improve the stability of learning, get rid of problems like mode collapse, and provide meaningful learning curves useful for debugging and hyperparameter searches. Furthermore, we show that the corresponding optimization problem is sound, and provide extensive theoretical work highlighting the deep connections to other distances between distributions.

[[Paper]](https://arxiv.org/abs/1701.07875) [[Code]](./src/wgan/wgan.py)

#### Example Running (WGAN)

```
$ cd src/wgan/
$ python3 wgan.py
```

### Wasserstein GAN GP

_Improved training of Wasserstein GANs_

#### Authors (WGAN GP)

Ishaan Gulrajani, Faruk Ahmed, Martin Arjovsky, Vincent Dumoulin, Aaron C. Courville

#### Abstract (WGAN GP)

Generative Adversarial Networks (GANs) are powerful generative models, but suffer from training instability. The recently proposed Wasserstein GAN (WGAN) makes progress toward stable training of GANs, but sometimes can still generate only poor samples or fail to converge. We find that these problems are often due to the use of weight clipping in WGAN to enforce a Lipschitz constraint on the critic, which can lead to undesired behavior. We propose an alternative to clipping weights: penalize the norm of gradient of the critic with respect to its input. Our proposed method performs better than standard WGAN and enables stable training of a wide variety of GAN architectures with almost no hyperparameter tuning, including 101-layer ResNets and language models with continuous generators. We also achieve high quality generations on CIFAR-10 and LSUN bedrooms.

[[Paper]](http://papers.nips.cc/paper/7159-improved-training-of-wasserstein-gans) [[Code]](./src/wgan_gp/wgan_gp.py)

#### Example Running (WGAN GP)

```
$ cd src/wgan_gp/
$ python3 wgan_gp.py
```

### UNIT

_Unsupervised Image-to-Image Translation Networks_

#### Authors (UNIT)

Ming-Yu Liu, Thomas Breuel, Jan Kautz

#### Abstract (UNIT)

Unsupervised image-to-image translation aims at learning a joint distribution of images in different domains by using images from the marginal distributions in individual domains. Since there exists an infinite set of joint distributions that can arrive the given marginal distributions, one could infer nothing about the joint distribution from the marginal distributions without additional assumptions. To address the problem, we make a shared-latent space assumption and propose an unsupervised image-to-image translation framework based on Coupled GANs. We compare the proposed framework with competing approaches and present high quality image translation results on various challenging unsupervised image translation tasks, including street scene image translation, animal image translation, and face image translation. We also apply the proposed framework to domain adaptation and achieve state-of-the-art performance on benchmark datasets. Code and additional results are available in this [https URL](https://github.com/mingyuliutw/unit).

[[Paper]](https://arxiv.org/abs/1703.00848) [[Code]](./src/unit/unit.py)

#### Example Running (UNIT)

```
$ cd data/
$ ./download_cyclegan_dataset.sh apple2orange
$ src/unit/
$ python3 unit.py --dataset_name apple2orange
```

### WaveGAN

_Adversarial Audio Synthesis_

#### Authors (WaveGAN)

Chris Donahue, Julian McAuley, Miller Puckette

#### Abstract (WaveGAN)

Audio signals are sampled at high temporal resolutions, and learning to synthesize audio requires capturing structure across a range of timescales. Generative adversarial networks (GANs) have seen wide success at generating images that are both locally and globally coherent, but they have seen little application to audio generation. In this paper we introduce WaveGAN, a first attempt at applying GANs to unsupervised synthesis of raw-waveform audio. WaveGAN is capable of synthesizing one second slices of audio waveforms with global coherence, suitable for sound effect generation. Our experiments demonstrate that, without labels, WaveGAN learns to produce intelligible words when trained on a small-vocabulary speech dataset, and can also synthesize audio from other domains such as drums, bird vocalizations, and piano. We compare WaveGAN to a method which applies GANs designed for image generation on image-like audio feature representations, finding both approaches to be promising.

[[Paper](https://arxiv.org/abs/1802.04208)] [[Code](./src/wavegan/wavegan.py)]

#### Example Running (WaveGAN)

Before running the WaveGAN, you should first download dataset in the data directory.

* `sc09`: [sc09 raw WAV files](http://deepyeti.ucsd.edu/cdonahue/sc09.tar.gz), utterances of spoken english words '0'-'9'
* `piano`: [Piano raw WAV files](http://deepyeti.ucsd.edu/cdonahue/mancini_piano.tar.gz)

After downloading the dataset, you could run the WaveGAN by using following scripts.

```
$ cd src/wavegan/
$ python3 train.py
```

#### Contributions (WaveGAN)

Codes for the WaveGAN in this repository are based on [mazzzystar's repository](https://github.com/mazzzystar/WaveGAN-pytorch) and [jtcramar's repository](https://github.com/jtcramer/wavegan).
Credits for the WaveGAN codes to [mazzzystar](https://github.com/mazzzystar) and [jtcramar](https://github.com/jtcramer).
