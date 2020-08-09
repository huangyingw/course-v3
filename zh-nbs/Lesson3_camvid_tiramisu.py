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

# # Lesson 3_camvid_tiramisu

# # Image segmentation with CamVid
# # 用CamVid数据集进行图像分割

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai import *
from fastai.vision import *
from fastai.callbacks.hooks import *

# The One Hundred Layer Tiramisu paper used a modified version of Camvid, with smaller images and few classes. You can get it from the CamVid directory of this repo:<br>
# One Hundred Layer Tiramisu这篇论文使用了改进版的CamVid数据集，该数据集图片更小、类别更少。你可以在以下库中的CamVid目录里找到它：
#
#     git clone https://github.com/alexgkendall/SegNet-Tutorial.git

path = Path('./data/camvid-tiramisu')

path.ls()

# ## Data
# ## 数据

fnames = get_image_files(path/'val')
fnames[:3]

lbl_names = get_image_files(path/'valannot')
lbl_names[:3]

img_f = fnames[0]
img = open_image(img_f)
img.show(figsize=(5,5))


# +
def get_y_fn(x): return Path(str(x.parent)+'annot')/x.name

codes = array(['Sky', 'Building', 'Pole', 'Road', 'Sidewalk', 'Tree',
    'Sign', 'Fence', 'Car', 'Pedestrian', 'Cyclist', 'Void'])
# -

mask = open_mask(get_y_fn(img_f))
mask.show(figsize=(5,5), alpha=1)

src_size = np.array(mask.shape[1:])
src_size,mask.data

# ## Datasets
# ## 数据集

bs,size = 8,src_size//2

src = (SegmentationItemList.from_folder(path)
       .split_by_folder(valid='val')
       .label_from_func(get_y_fn, classes=codes))

data = (src.transform(get_transforms(), tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

data.show_batch(2, figsize=(10,7))

# ## Model
# ## 模型

# +
name2id = {v:k for k,v in enumerate(codes)}
void_code = name2id['Void']

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != void_code
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()


# -

metrics=acc_camvid
wd=1e-2

learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd, bottle=True)

lr_find(learn)
learn.recorder.plot()

lr=2e-3

learn.fit_one_cycle(10, slice(lr), pct_start=0.8)

learn.save('stage-1')

learn.load('stage-1');

learn.unfreeze()

lrs = slice(lr/100,lr)

learn.fit_one_cycle(12, lrs, pct_start=0.8)

learn.save('stage-2');

# ## Go big
# ## 用更大的数据集进行训练

learn=None
gc.collect()

# You may have to restart your kernel and come back to this stage if you run out of memory, and may also need to decrease `bs`.<br>
# 如果内存不够的话，你可能需要重启你的计算内核再返回这一步，同时可能要减少`bs`的设定。

size = src_size
bs=8

data = (src.transform(get_transforms(), size=size, tfm_y=True)
        .databunch(bs=bs)
        .normalize(imagenet_stats))

learn = unet_learner(data, models.resnet34, metrics=metrics, wd=wd, bottle=True).load('stage-2');

lr_find(learn)
learn.recorder.plot()

lr=1e-3

learn.fit_one_cycle(10, slice(lr), pct_start=0.8)

learn.save('stage-1-big')

learn.load('stage-1-big');

learn.unfreeze()

lrs = slice(lr/1000,lr/10)

learn.fit_one_cycle(10, lrs)

learn.save('stage-2-big')

learn.load('stage-2-big');

learn.show_results(rows=3, figsize=(9,11))

# ## fin 
# ## 小结

# +
# start: 480x360
# -

print(learn.summary())
