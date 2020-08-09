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

from fastai.vision import *

# ## Data

path = Config().data_path()/'imagenet'

# ### Inputs: precomputed activations

# We will build a model on the whole of ImageNet, so we will compute once and for all the activations for the whole training and validation set. We use `presize` to set the images to 224 x 224 by just using PIL.

src = (ImageList.from_folder(path)
          .split_by_folder()
          .label_from_folder())

data = src.presize(size=224).databunch(bs=256).normalize(imagenet_stats)

# This is the pretrained resnet50 with concat pool and flatten:

body = create_body(models.resnet50)
layers = list(body.children())
layers += [AdaptiveConcatPool2d(), Flatten()]   
body = nn.Sequential(*layers).to(defaults.device)

# We will use bcolz to store our activations in an array that's saved to memory (all won't fit in RAM). Install with 
# ```
# pip install -U bcolz
# ```

import bcolz

tmp_path = path/'tmp'


# +
#To clean-up previous tries
#shutil.rmtree(tmp_path)
# -

# Those functions will store the precomputed activations in `tmp_path`.

def precompute_activations_dl(dl, model, path:Path, force:bool=False):
    model.eval()
    if os.path.exists(path) and not force: return
    arr = bcolz.carray(np.zeros((0,4096), np.float32), chunklen=1, mode='w', rootdir=path)
    with torch.no_grad():
        for x,y in progress_bar(dl):
            z = model(x)
            arr.append(z.cpu().numpy())
            arr.flush()


def precompute_activations(data, model, path:Path, force:bool=False):
    os.makedirs(path, exist_ok=True)
    precompute_activations_dl(data.fix_dl,   model, path/'train', force=force) #Use fix_dl and not train_dl for shuffle=False
    precompute_activations_dl(data.valid_dl, model, path/'valid', force=force)


precompute_activations(data, body, tmp_path)

# Save the labels and the filenames in the same order as our activations.

np.save(tmp_path/'trn_lbl.npy', data.train_ds.y.items)
np.save(tmp_path/'val_lbl.npy', data.valid_ds.y.items)
save_texts(tmp_path/'classes.txt', data.train_ds.classes)

np.save(tmp_path/'trn_names.npy', data.train_ds.x.items)
np.save(tmp_path/'val_names.npy', data.valid_ds.x.items)


# To load our precomputed activations, we'll use the following `ItemList`

class BcolzItemList(ItemList):
    def __init__(self, path, **kwargs):
        self.arr = bcolz.open(path)
        super().__init__(range(len(self.arr)), **kwargs)
    
    def get(self, i): return self.arr[i]


src = ItemLists(path, BcolzItemList(path/'tmp'/'train'), BcolzItemList(path/'tmp'/'valid'))

# ### Targets: word vectors

# We build a regression model that has to predict a vector from the image features. We need to associate a word vector to each one of our 1000 classes.

classes = loadtxt_str(tmp_path/'classes.txt')

classes, len(classes)

# The labels in imagenet are codes that come from [wordnet](https://wordnet.princeton.edu/). So let's download the corresponding dictionary.

WORDNET = 'classids.txt'
download_url(f'http://files.fast.ai/data/{WORDNET}', path/'tmp'/WORDNET)

class_ids = loadtxt_str(path/'tmp'/WORDNET)
class_ids = dict([l.strip().split() for l in class_ids])

named_classes = [class_ids[c] for c in classes]
named_classes[:10]

# We will train our model to predict not the label of its class, but the corresponding pretrained vector. There are plenty of word embeddings available, here we will use fastText.
#
# To install fastText:
# ```
# $ git clone https://github.com/facebookresearch/fastText.git
# $ cd fastText
# $ pip install .
# ```
#
# To download the english embeddings:
#
# ```
# $ wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
# ```

import fasttext as ft
en_vecs = ft.load_model(str((path/'cc.en.300.bin')))

# A lot of our classes are actually composed of several words separated by a `_`. The pretrained word vectors from fastText won't know them directly, but it can still compute a word vector to represent them:

vec_dog = en_vecs.get_sentence_vector('dog')
vec_lab = en_vecs.get_sentence_vector('labrador')
vec_gor = en_vecs.get_sentence_vector('golden retriever')
vec_ban = en_vecs.get_sentence_vector('banana')

# To check if two word vectors are close or not, we use cosine similarity.

F.cosine_similarity(tensor(vec_dog[None]), tensor(vec_lab[None]))

F.cosine_similarity(tensor(vec_dog[None]), tensor(vec_ban[None]))

F.cosine_similarity(tensor(vec_lab[None]), tensor(vec_ban[None]))

F.cosine_similarity(tensor(vec_dog[None]), tensor(vec_gor[None]))

F.cosine_similarity(tensor(vec_lab[None]), tensor(vec_gor[None]))

# So let's grab all the word vectors for all our classes:

vecs = []
for n in named_classes:
    vecs.append(en_vecs.get_sentence_vector(n.replace('_', ' ')))

# Then we label each feature map with the word vector of its target.

train_labels = np.load(tmp_path/'trn_lbl.npy')
valid_labels = np.load(tmp_path/'val_lbl.npy')
train_vecs = [vecs[l] for l in train_labels]
valid_vecs = [vecs[l] for l in valid_labels]

# We use our custom `BcolzItemList` to gather the data:

src = ItemLists(path, BcolzItemList(tmp_path/'train'), BcolzItemList(tmp_path/'valid'))
src = src.label_from_lists(train_vecs, valid_vecs, label_cls=FloatList)

data = src.databunch(bs=512, num_workers=16)

model = create_head(4096, data.c, lin_ftrs = [1024], ps=[0.2,0.2])
model = nn.Sequential(*list(model.children())[2:])


def cos_loss(inp,targ): return 1 - F.cosine_similarity(inp,targ).mean()


learn = Learner(data, model, loss_func=cos_loss)

learn.lr_find()

learn.recorder.plot()

learn.fit_one_cycle(15,3e-2)

learn.save('fit')

learn.model.eval()
preds = []
with torch.no_grad():
    for x,y in progress_bar(learn.data.fix_dl):
        preds.append(learn.model(x).cpu().numpy())
    for x,y in progress_bar(learn.data.valid_dl):
        preds.append(learn.model(x).cpu().numpy())

preds = np.concatenate(preds, 0)

np.save(path/'preds.npy', preds)

# ### Looking at predicted tags in image classes

# Now we will check, for one given image, what are the word vectors that are the closes to it. To compute this very quickly, we use `nmslib` which is very fast (pip install nmslib).

# +
import nmslib

def create_index(a):
    index = nmslib.init(space='angulardist')
    index.addDataPointBatch(a)
    index.createIndex()
    return index

def get_knns(index, vecs):
     return zip(*index.knnQueryBatch(vecs, k=10, num_threads=4))

def get_knn(index, vec): return index.knnQuery(vec, k=10)


# -

# We first look in the word vectors of our given classes:

nn_classes = create_index(vecs)

valid_preds = preds[-len(data.valid_ds):]
valid_names = np.load(tmp_path/'val_names.npy')

idxs,dists = get_knns(nn_classes, valid_preds)

ks = [0,10000,20000,30000]
_,axs = plt.subplots(2,2,figsize=(12,8))
for k,ax in zip(ks, axs.flatten()):
    open_image(valid_names[k]).show(ax = ax)
    title = ','.join([class_ids[classes[i]] for i in idxs[k][:3]])
    title += f'\n{class_ids[classes[valid_labels[k]]]}'
    ax.set_title(title)

# ### Looking at predicted tags in all Wordnet

# Now let's look at the words it finds in all Wordnet.

words,wn_vecs = [],[]
for k,n in class_ids.items():
    words.append(n)
    wn_vecs.append(en_vecs.get_sentence_vector(n.replace('_', ' ')))

nn_wvs = create_index(wn_vecs)

idxs,dists = get_knns(nn_wvs, valid_preds)

ks = [0,10000,20000,30000]
_,axs = plt.subplots(2,2,figsize=(12,8))
for k,ax in zip(ks, axs.flatten()):
    open_image(valid_names[k]).show(ax = ax)
    title = ','.join([words[i] for i in idxs[k][:3]])
    title += f'\n{class_ids[classes[valid_labels[k]]]}'
    ax.set_title(title)

# ### Text -> Image search

# We can use the reverse approach: feed a word vector and find the image activations that match it the closest:

nn_preds = create_index(valid_preds)


def show_imgs_from_text(text):
    vec = en_vecs.get_sentence_vector(text)
    idxs,dists = get_knn(nn_preds, vec)
    _,axs = plt.subplots(2,2,figsize=(12,8))
    for i,ax in zip(idxs[:4], axs.flatten()):
        open_image(valid_names[i]).show(ax = ax)


# 'boat' isn't a label in ImageNet, yet if we ask the images whose vord vectors are the most similar to the word vector for boat...

show_imgs_from_text('boat')

# or even more precisely 'motor boat'

show_imgs_from_text('motor boat')

show_imgs_from_text('sail boat')

# ### Image->image

# We can also ask for the images with a word vector most similar to another image. This one was downloaded from Google and isn't in Imagenet.

img = open_image('images/teddy_bear.jpg')
img

# To get the corresdponding vector, we need to feed it to the pretrained model (`body`, defined at the top) after normalizing it.

img = img.data
m,s = imagenet_stats
x = (img - tensor(m)[:,None,None])/tensor(s)[:,None,None]

activs = body.eval()(x[None].cuda())

pred = learn.model.eval()(activs)

pred = pred[0].detach().cpu().numpy()

idxs,dists = get_knn(nn_preds, pred)
_,axs = plt.subplots(2,2,figsize=(12,8))
for i,ax in zip(idxs[:4], axs.flatten()):
    open_image(valid_names[i]).show(ax = ax)


