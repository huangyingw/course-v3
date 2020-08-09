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

# # Lesson6_pets_more

# # pets revisited
# # 宠物分类问题回顾

# +
# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai.vision import *
# -

bs = 64

path = untar_data(URLs.PETS)/'images'

# ## Data augmentation 数据增强

tfms = get_transforms(max_rotate=20, max_zoom=1.3, max_lighting=0.4, max_warp=0.4,
                      p_affine=1., p_lighting=1.)

doc(get_transforms)

src = ImageList.from_folder(path).split_by_rand_pct(0.2, seed=2)


def get_data(size, bs, padding_mode='reflection'):
    return (src.label_from_re(r'([^/]+)_\d+.jpg$')
           .transform(tfms, size=size, padding_mode=padding_mode)
           .databunch(bs=bs).normalize(imagenet_stats))


data = get_data(224, bs, 'zeros')


# +
def _plot(i,j,ax):
    x,y = data.train_ds[3]
    x.show(ax, y=y)

plot_multi(_plot, 3, 3, figsize=(8,8))
# -

data = get_data(224,bs)

plot_multi(_plot, 3, 3, figsize=(8,8))

# ## Train a model 训练网络

gc.collect()
learn = cnn_learner(data, models.resnet34, metrics=error_rate, bn_final=True)

learn.fit_one_cycle(3, slice(1e-2), pct_start=0.8)

learn.unfreeze()
learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-3), pct_start=0.8)

data = get_data(352,bs)
learn.data = data

learn.fit_one_cycle(2, max_lr=slice(1e-6,1e-4))

learn.save('352')

# ## Convolution kernel 卷积核

data = get_data(352,16)

learn = cnn_learner(data, models.resnet34, metrics=error_rate, bn_final=True).load('352')

idx=0
x,y = data.valid_ds[idx]
x.show()
data.valid_ds.y[idx]

k = tensor([
    [0.  ,-5/3,1],
    [-5/3,-5/3,1],
    [1.  ,1   ,1],
]).expand(1,3,3,3)/6

k

k.shape

t = data.valid_ds[0][0].data; t.shape

t[None].shape

edge = F.conv2d(t[None], k)

show_image(edge[0], figsize=(5,5));

data.c

learn.model

print(learn.summary())

# ## Heatmap 热力图

m = learn.model.eval();

xb,_ = data.one_item(x)
xb_im = Image(data.denorm(xb)[0])
xb = xb.cuda()

from fastai.callbacks.hooks import *


def hooked_backward(cat=y):
    with hook_output(m[0]) as hook_a: 
        with hook_output(m[0], grad=True) as hook_g:
            preds = m(xb)
            preds[0,int(cat)].backward()
    return hook_a,hook_g


hook_a,hook_g = hooked_backward()

acts  = hook_a.stored[0].cpu()
acts.shape

avg_acts = acts.mean(0)
avg_acts.shape


def show_heatmap(hm):
    _,ax = plt.subplots()
    xb_im.show(ax)
    ax.imshow(hm, alpha=0.6, extent=(0,352,352,0),
              interpolation='bilinear', cmap='magma');


show_heatmap(avg_acts)

# ## Grad-CAM

# Paper: [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391) <br><br>
# 论文: [Grad-CAM: 基于神经网络梯度局部化的可视化解释](https://arxiv.org/abs/1610.02391)

grad = hook_g.stored[0][0].cpu()
grad_chan = grad.mean(1).mean(1)
grad.shape,grad_chan.shape

mult = (acts*grad_chan[...,None,None]).mean(0)

show_heatmap(mult)

fn = path/'../other/bulldog_maine.jpg' #Replace with your own image

x = open_image(fn); x

xb,_ = data.one_item(x)
xb_im = Image(data.denorm(xb)[0])
xb = xb.cuda()

hook_a,hook_g = hooked_backward()

# +
acts = hook_a.stored[0].cpu()
grad = hook_g.stored[0][0].cpu()

grad_chan = grad.mean(1).mean(1)
mult = (acts*grad_chan[...,None,None]).mean(0)
# -

show_heatmap(mult)

data.classes[0]

hook_a,hook_g = hooked_backward(0)

# +
acts = hook_a.stored[0].cpu()
grad = hook_g.stored[0][0].cpu()

grad_chan = grad.mean(1).mean(1)
mult = (acts*grad_chan[...,None,None]).mean(0)
# -

show_heatmap(mult)
