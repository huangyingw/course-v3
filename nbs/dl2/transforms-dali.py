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

# # Data augmentation

# +
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline
# -

#export
from exp.nb_09 import *

# ## PIL transforms

path = datasets.untar_data(datasets.URLs.IMAGENETTE_160)
tfms = [make_rgb, ResizeFixed(128), to_byte_tensor, to_float_tensor]

from nvidia.dali.pipeline import Pipeline

import nvidia.dali.ops as ops
import nvidia.dali.types as types

bs=8


class SimplePipeline(Pipeline):
    def __init__(self, batch_size=8, num_threads=8, device_id=0):
        super(SimplePipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = path/'train')
        # self.input = ops.FileReader(file_root = image_dir, file_list = image_dir + '/file_list.txt')
        self.resize = ops.Resize(device = "cpu", resize_x=128, resize_y=128, image_type = types.RGB,
            interp_type = types.INTERP_LINEAR)
        self.decode = ops.HostDecoder(output_type = types.RGB)

    def define_graph(self):
        jpegs, labels = self.input(name='r')
        images = self.decode(jpegs)
        images = self.resize(images)
        return (images, labels)


pipe = SimplePipeline()
pipe.build()

pipe_out = pipe.run()
print(pipe_out)

images, labels = pipe_out
images.is_dense_tensor(), labels.is_dense_tensor()

t = images.as_tensor()

from nvidia.dali.plugin.pytorch import DALIGenericIterator,DALIClassificationIterator,feed_ndarray

it = DALIGenericIterator(pipe, ['data','label'], pipe.epoch_size('r'))

it = DALIClassificationIterator(pipe, pipe.epoch_size('r'))

its = iter(it)

t = next(it)[0]

t['label'].cuda().long().type()

t['data'].type()

import numpy as np

labels_tensor = labels.as_tensor()

labels_tensor.shape()

np.array(labels_tensor)

import matplotlib.gridspec as gridspec


def show_images(image_batch):
    columns = 4
    rows = len(image_batch) // (columns)
    fig = plt.figure(figsize = (32,(32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows*columns):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(image_batch.at(j))


show_images(images)


