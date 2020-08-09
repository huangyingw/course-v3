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

# +
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline
# -

from exp.nb_08 import *

path = datasets.untar_data(datasets.URLs.IMAGENETTE_320)

# +
tfms = [make_rgb, ResizeFixed(224), to_byte_tensor, to_float_tensor]

il = ImageList.from_files(path, tfms=tfms)
sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val'))
ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())
# -

bs=256

train_dl,valid_dl = get_dls(ll.train,ll.valid,bs, num_workers=4)

# %time x,y = next(iter(train_dl))

# %time for x,y in train_dl: x,y = x.cuda(),y.cuda()


