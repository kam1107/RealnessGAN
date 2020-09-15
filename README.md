# RealnessGAN
This repository contains the code of the following paper:
> **Real or Not Real, that is the Question**<br>
> Yuanbo Xiangli*, Yubin Deng*, Bo Dai*, Chen Change Loy, Dahua Lin<br>
> [paper](https://openreview.net/forum?id=B1lPaCNtPB) & [talk](https://youtu.be/ddYLx6kqcMg)
>
> **Abstract:** *While generative adversarial networks (GAN) have been widely adopted in various topics, in this paper we generalize the standard GAN to a new perspective by treating realness as a random variable that can be estimated from multiple angles. In this generalized framework, referred to as RealnessGAN, the discriminator outputs a distribution as the measure of realness. While RealnessGAN shares similar theoretical guarantees with the standard GAN, it provides more insights on adversarial learning. More importantly, compared to multiple baselines, RealnessGAN provides stronger guidance for the generator, achieving improvements on both synthetic and real-world datasets. Moreover, it enables the basic DCGAN architecture to generate realistic images at 1024*1024 resolution when trained from scratch.

## Demo

* Watch the full demo video in [YouTube](https://www.youtube.com/watch?v=ddYLx6kqcMg)

## Dataset
Experiments were conducted on two real-world datasets: [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [FFHQ](https://github.com/NVlabs/ffhq-dataset); and a toy dataset: [Mixture of Gaussians](/data/MixtureGaussian3By3.pk). 

## Requirements
* Python 3.6
* Pytorch 1.1.0

## Pretrain Models

* [CelebA](https://drive.google.com/file/d/1GObH02xlPQPnmAhCVIzMOK7LyqOpZn59/view?usp=sharing)
* [FFHQ](https://drive.google.com/file/d/1NapKYf90NMQwk0TNTuMIb22RN3OeC0PU/view?usp=sharing)

## Training
* Either use the aforementioned dataset or prepare your dataset.
* Scripts to run experiments are stored in /scripts/*.sh.
* Edit folder locations in your scripts. Make sure the folders to store LOG and OUTPUT are created.
* Run `./scripts/run_your_scripts.sh`

## Snapshots

**CelebA 256x256 (FID = 23.51)**

![](/images/CelebA_snapshot.png)

**FFHQ 1024x1024 (FID = 17.18)**

![](/images/FFHQ_snapshot.png)

## Bibtex

```
@article{xiangli2020real,
  title={Real or Not Real, that is the Question},
  author={Xiangli, Yuanbo and Deng, Yubin and Dai, Bo and Loy, Chen Change and Lin, Dahua},
  booktitle={International Conference on Learning Representations},
  year={2020}
}
```


