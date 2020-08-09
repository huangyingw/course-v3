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

# # Lesson3_head_pose

# # Regression with BIWI head pose dataset<br> 
# # 用BIWI头部姿势数据集进行回归建模

# This is a more advanced example to show how to create custom datasets and do regression with images. Our task is to find the center of the head in each image. The data comes from the [BIWI head pose dataset](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html#db), thanks to Gabriele Fanelli et al. We have converted the images to jpeg format, so you should download the converted dataset from [this link](https://s3.amazonaws.com/fast-ai-imagelocal/biwi_head_pose.tgz).<br>

# 这个案例是一个更高级的示例，它展示了如何创建自定义数据集，并且对图像进行回归建模。 我们的任务是在每个图片中确定头部的中心位置。数据来自[BIWI头部姿势数据集](https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html#db)。感谢Gabriele Fanelli等人的努力。我们已经把图片转化为jpeg格式，因此你应该从[这里](https://s3.amazonaws.com/fast-ai-imagelocal/biwi_head_pose.tgz)下载转化好的数据。

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai.vision import *

# ## Getting and converting the data
# ## 数据获取与格式转换

path = untar_data(URLs.BIWI_HEAD_POSE)

cal = np.genfromtxt(path/'01'/'rgb.cal', skip_footer=6); cal

fname = '09/frame_00667_rgb.jpg'


def img2txt_name(f): return path/f'{str(f)[:-7]}pose.txt'


img = open_image(path/fname)
img.show()

ctr = np.genfromtxt(img2txt_name(fname), skip_header=3); ctr


# +
def convert_biwi(coords):
    c1 = coords[0] * cal[0][0]/coords[2] + cal[0][2]
    c2 = coords[1] * cal[1][1]/coords[2] + cal[1][2]
    return tensor([c2,c1])

def get_ctr(f):
    ctr = np.genfromtxt(img2txt_name(f), skip_header=3)
    return convert_biwi(ctr)

def get_ip(img,pts): return ImagePoints(FlowField(img.size, pts), scale=True)


# -

get_ctr(fname)

ctr = get_ctr(fname)
img.show(y=get_ip(img, ctr), figsize=(6, 6))

# ## Creating a dataset
# ## 创建一个数据集

data = (PointsItemList.from_folder(path)
        .split_by_valid_func(lambda o: o.parent.name=='13')
        .label_from_func(get_ctr)
        .transform(get_transforms(), tfm_y=True, size=(120,160))
        .databunch().normalize(imagenet_stats)
       )

data.show_batch(3, figsize=(9,6))

# ## Train model
# ## 训练模型

learn = cnn_learner(data, models.resnet34)

learn.lr_find()
learn.recorder.plot()

lr = 2e-2

learn.fit_one_cycle(5, slice(lr))

learn.save('stage-1')

learn.load('stage-1');

learn.show_results()

# ## Data augmentation
# ## 数据增强

# +
tfms = get_transforms(max_rotate=20, max_zoom=1.5, max_lighting=0.5, max_warp=0.4, p_affine=1., p_lighting=1.)

data = (PointsItemList.from_folder(path)
        .split_by_valid_func(lambda o: o.parent.name=='13')
        .label_from_func(get_ctr)
        .transform(tfms, tfm_y=True, size=(120,160))
        .databunch().normalize(imagenet_stats)
       )


# +
def _plot(i,j,ax):
    x,y = data.train_ds[0]
    x.show(ax, y=y)

plot_multi(_plot, 3, 3, figsize=(8,6))
