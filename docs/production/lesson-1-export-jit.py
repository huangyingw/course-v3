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

# # fast.ai lesson 1 - training on Notebook Instance and export to torch.jit model

# ## Overview
# This notebook shows how to use the SageMaker Python SDK to train your fast.ai model on a SageMaker notebook instance then export it as a torch.jit model to be used for inference on AWS Lambda.
#
# ## Set up the environment
#
# You will need a Jupyter notebook with the `boto3` and `fastai` libraries installed. You can do this with the command `pip install boto3 fastai`
#
# This notebook was created and tested on a single ml.p3.2xlarge notebook instance. 
#

# ## Train your model
#
# We are going to train a fast.ai model as per [Lesson 1 of the fast.ai MOOC course](https://course.fast.ai/videos/?lesson=1) locally on the SageMaker Notebook instance. We will then save the model weights and upload them to S3.
#
#

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# +
import os
import io
import tarfile

import PIL

import boto3

from fastai.vision import *
# -

path = untar_data(URLs.PETS); path

path_anno = path/'annotations'
path_img = path/'images'
fnames = get_image_files(path_img)
np.random.seed(2)
pat = re.compile(r'/([^/]+)_\d+.jpg$')

bs=64
img_size=299

data = ImageDataBunch.from_name_re(path_img, fnames, pat, ds_tfms=get_transforms(),
                                   size=img_size, bs=bs//2).normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet50, metrics=error_rate)

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(8)

learn.unfreeze()
learn.fit_one_cycle(3, max_lr=slice(1e-6,1e-4))

# ## Export model and upload to S3

# Now that we have trained our model we need to export it, create a tarball of the artefacts and upload to S3.
#

# First we need to export the class names from the data object into a text file.

save_texts(path_img/'models/classes.txt', data.classes)

# Now we need to export the model in the [PyTorch TorchScript format](https://pytorch.org/docs/stable/jit.html) so we can load into an AWS Lambda function.

trace_input = torch.ones(1,3,img_size,img_size).cuda()
jit_model = torch.jit.trace(learn.model.float(), trace_input)
model_file='resnet50_jit.pth'
output_path = str(path_img/f'models/{model_file}')
torch.jit.save(jit_model, output_path)

# Next step is to create a tarfile of the exported classes file and model weights.

tar_file=path_img/'models/model.tar.gz'
classes_file='classes.txt'

with tarfile.open(tar_file, 'w:gz') as f:
    f.add(path_img/f'models/{model_file}', arcname=model_file)
    f.add(path_img/f'models/{classes_file}', arcname=classes_file)

# Now we need to upload the model tarball to S3.

s3 = boto3.resource('s3')
s3.meta.client.upload_file(str(tar_file), 'REPLACE_WITH_YOUR_BUCKET_NAME', 'fastai-models/lesson1/model.tar.gz')
