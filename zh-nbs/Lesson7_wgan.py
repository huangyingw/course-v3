# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Practical Deep Learning for Coders, v3

# # Lesson7_wgan

# # wgan

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai.vision import *
from fastai.vision.gan import *

# ## LSun bedroom data

# ## 起居室数据集

# For this lesson, we'll be using the bedrooms from the LSUN dataset. The full dataset is a bit too large so we'll use a sample from kaggle.

# 本节课，我们将使用[LSUN dataset](http://lsun.cs.princeton.edu/2017/)的起居室数据集。整个数据集略大，因此这里我们只从[kaggle](https://www.kaggle.com/jhoward/lsun_bedroom)节选部分数据。

path = untar_data(URLs.LSUN_BEDROOMS)


# We then grab all the images in the folder with the data block API. We don't create a validation set here for reasons we'll explain later. It consists of random noise of size 100 by default (can be changed below) as inputs and the images of bedrooms as targets. That's why we do tfm_y=True in the transforms, then apply the normalization to the ys and not the xs.

# 我们使用data block API抓取所有图片到文件夹中。我们不会创建一个独立验证集，原因后面我们会做解释。输入数据里包含了均值为100的随机噪声（也可以修改），并将起居室的图片作为目标。这也是我们在转换时，使用`tfm_y=True`参数，并且对ys而不是xs执行正规化操作的原因。

def get_data(bs, size):
    return (GANItemList.from_folder(path, noise_sz=100)
               .split_none()
               .label_from_func(noop)
               .transform(tfms=[[crop_pad(size=size, row_pct=(0,1), col_pct=(0,1))], []], size=size, tfm_y=True)
               .databunch(bs=bs)
               .normalize(stats = [torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])], do_x=False, do_y=True))


# We'll begin with a small side and use gradual resizing.

# 我们将从小规模开始，逐步调整规模。

data = get_data(128, 64)

data.show_batch(rows=5)

# ## Models 模型

# GAN stands for Generative Adversarial Nets and were invented by Ian Goodfellow. The concept is that we will train two models at the same time: a generator and a critic. The generator will try to make new images similar to the ones in our dataset, and the critic will try to classify real images from the ones the generator does. The generator returns images, the critic a single number (usually 0. for fake images and 1. for real ones).

# GAN即生成对抗网络 [Generative Adversarial Nets](https://arxiv.org/pdf/1406.2661.pdf)，该网络由Ian Goodfellow发明。其核心概念是我们将同时训练两个模型：一个生成器网络和一个评判网络。生成器网络将自创和数据集中的图片相似的新图片，而评判网络则试图从生成器生成的图片中分辨出真实的图片。生成器给出图片，评判网络则输出一个（作为评判结果的）值（通常0表示赝品,1表示真品）。

# We train them against each other in the sense that at each step (more or less), we:

# 这两个网络每一步都在做对抗训练（或多或少可以这么解释），我们（如下这样操做）：

# 1.	Freeze the generator and train the critic for one step by:
#     -	getting one batch of true images (let's call that real)
#     -	generating one batch of fake images (let's call that fake)
#     -	have the critic evaluate each batch and compute a loss function from that; the important part is that it rewards positively the detection of real images and penalizes the fake ones
#     -	update the weights of the critic with the gradients of this loss
#

# 1. 冻结生成器训练评判网络:
#   - 获取一批真品图片(我们简称为`真`)
#   - 产生一批赝品图片(我们简称为`假`)
#   - 让评判网络评估这批数据并计算出损失函数值；重要的一点是，如果判断为真就要给与奖励，反之，判断为假则予以惩罚。
#   - 使用损失函数的梯度来更新评判网络的权重

# 2.	Freeze the critic and train the generator for one step by:
#     - generating one batch of fake images
#     - evaluate the critic on it
#     - return a loss that rewards posisitivly the critic thinking those are real images; the important part is that it rewards positively the detection of real images and penalizes the fake ones
#     - update the weights of the generator with the gradients of this loss

# 2. 冻结评判网络训练生成器：
#   - 产生一批赝品图片
#   - 评判网络对这些图片进行评判
#   - 返回损失（函数），这是评判网络得到的所有真实图片的回馈，它的重要作用是，探测出真实图片就给予奖励,反之则给予惩罚
#   - 使用损失函数的梯度来更新生成器网络的权重

# Here, we'll use the Wassertein GAN.

# 这里,我们将使用[Wassertein GAN](https://arxiv.org/pdf/1701.07875.pdf)这篇论文中的GAN网络。

# We create a generator and a critic that we pass to `gan_learner`. The noise_size is the size of the random vector from which our generator creates images.<br><br>
# 我们将创建一个生成器和一个评判网络，把他们作为参数传给`gan_learner`。noise_size参数是生成器创造新图片时的随机向量的变动范围。

generator = basic_generator(in_size=64, n_channels=3, n_extra_layers=1)
critic    = basic_critic   (in_size=64, n_channels=3, n_extra_layers=1)

learn = GANLearner.wgan(data, generator, critic, switch_eval=False,
                        opt_func = partial(optim.Adam, betas = (0.,0.99)), wd=0.)

learn.fit(30,2e-4)

learn.gan_trainer.switch(gen_mode=True)
learn.show_results(ds_type=DatasetType.Train, rows=16, figsize=(8,8))
