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

# + [markdown] hide_input=false
# # Creating your own dataset from Google Images
#
# *by: Francisco Ingham and Jeremy Howard. Inspired by [Adrian Rosebrock](https://www.pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/)*

# + [markdown] hide_input=true
# In this tutorial we will see how to easily create an image dataset through Google Images. **Note**: You will have to repeat these steps for any new category you want to Google (e.g once for dogs and once for cats).

# + hide_input=false
from fastai.vision import *
# -

# ## Get a list of URLs

# ### Search and scroll

# Go to [Google Images](http://images.google.com) and search for the images you are interested in. The more specific you are in your Google Search, the better the results and the less manual pruning you will have to do.
#
# Scroll down until you've seen all the images you want to download, or until you see a button that says 'Show more results'. All the images you scrolled past are now available to download. To get more, click on the button, and continue scrolling. The maximum number of images Google Images shows is 700.
#
# It is a good idea to put things you want to exclude into the search query, for instance if you are searching for the Eurasian wolf, "canis lupus lupus", it might be a good idea to exclude other variants:
#
#     "canis lupus lupus" -dog -arctos -familiaris -baileyi -occidentalis
#
# You can also limit your results to show only photos by clicking on Tools and selecting Photos from the Type dropdown.

# ### Download into file

# Now you must run some Javascript code in your browser which will save the URLs of all the images you want for you dataset.
#
# In Google Chrome press <kbd>Ctrl</kbd><kbd>Shift</kbd><kbd>j</kbd> on Windows/Linux and <kbd>Cmd</kbd><kbd>Opt</kbd><kbd>j</kbd> on macOS, and a small window the javascript 'Console' will appear. In Firefox press <kbd>Ctrl</kbd><kbd>Shift</kbd><kbd>k</kbd> on Windows/Linux or <kbd>Cmd</kbd><kbd>Opt</kbd><kbd>k</kbd> on macOS. That is where you will paste the JavaScript commands.
#
# You will need to get the urls of each of the images. Before running the following commands, you may want to disable ad blocking extensions (uBlock, AdBlockPlus etc.) in Chrome. Otherwise the window.open() command doesn't work. Then you can run the following commands:
#
# ```javascript
# urls=Array.from(document.querySelectorAll('.rg_i')).map(el=> el.hasAttribute('data-src')?el.getAttribute('data-src'):el.getAttribute('data-iurl'));
# window.open('data:text/csv;charset=utf-8,' + escape(urls.join('\n')));
# ```

# ### Create directory and upload urls file into your server

# Choose an appropriate name for your labeled images. You can run these steps multiple times to create different labels.

folder = 'black'
file = 'urls_black.csv'

folder = 'teddys'
file = 'urls_teddys.csv'

folder = 'grizzly'
file = 'urls_grizzly.csv'

# You will need to run this cell once per each category.

path = Path('data/bears')
dest = path/folder
dest.mkdir(parents=True, exist_ok=True)

path.ls()

# Finally, upload your urls file. You just need to press 'Upload' in your working directory and select your file, then click 'Upload' for each of the displayed files.
#
# ![uploaded file](images/download_images/upload.png)

# ## Download images

# Now you will need to download your images from their respective urls.
#
# fast.ai has a function that allows you to do just that. You just have to specify the urls filename as well as the destination folder and this function will download and save all images that can be opened. If they have some problem in being opened, they will not be saved.
#
# Let's download our images! Notice you can choose a maximum number of images to be downloaded. In this case we will not download all the urls.
#
# You will need to run this line once for every category.

classes = ['teddys','grizzly','black']

download_images(path/file, dest, max_pics=200)

# If you have problems download, try with `max_workers=0` to see exceptions:
download_images(path/file, dest, max_pics=20, max_workers=0)

# Then we can remove any images that can't be opened:

for c in classes:
    print(c)
    verify_images(path/c, delete=True, max_size=500)

# ## View data

np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)

# +
# If you already cleaned your data, run this cell instead of the one before
# np.random.seed(42)
# data = ImageDataBunch.from_csv(path, folder=".", valid_pct=0.2, csv_labels='cleaned.csv',
#         ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
# -

# Good! Let's take a look at some of our pictures then.

data.classes

data.show_batch(rows=3, figsize=(7,8))

data.classes, data.c, len(data.train_ds), len(data.valid_ds)

# ## Train model

learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(4)

learn.save('stage-1')

learn.unfreeze()

learn.lr_find()

# If the plot is not showing try to give a start and end learning rate
# learn.lr_find(start_lr=1e-5, end_lr=1e-1)
learn.recorder.plot()

learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))

learn.save('stage-2')

# ## Interpretation

learn.load('stage-2');

interp = ClassificationInterpretation.from_learner(learn)

interp.plot_confusion_matrix()

# ## Cleaning Up
#
# Some of our top losses aren't due to bad performance by our model. There are images in our data set that shouldn't be.
#
# Using the `ImageCleaner` widget from `fastai.widgets` we can prune our top losses, removing photos that don't belong.

from fastai.widgets import *

# First we need to get the file paths from our top_losses. We can do this with `.from_toplosses`. We then feed the top losses indexes and corresponding dataset to `ImageCleaner`.
#
# Notice that the widget will not delete images directly from disk but it will create a new csv file `cleaned.csv` from where you can create a new ImageDataBunch with the corrected labels to continue training your model.

# In order to clean the entire set of images, we need to create a new dataset without the split. The video lecture demostrated the use of the `ds_type` param which no longer has any effect. See [the thread](https://forums.fast.ai/t/duplicate-widget/30975/10) for more details.

db = (ImageList.from_folder(path)
                   .split_none()
                   .label_from_folder()
                   .transform(get_transforms(), size=224)
                   .databunch()
     )

# +
# If you already cleaned your data using indexes from `from_toplosses`,
# run this cell instead of the one before to proceed with removing duplicates.
# Otherwise all the results of the previous step would be overwritten by
# the new run of `ImageCleaner`.

# db = (ImageList.from_csv(path, 'cleaned.csv', folder='.')
#                    .split_none()
#                    .label_from_df()
#                    .transform(get_transforms(), size=224)
#                    .databunch()
#      )
# -

# Then we create a new learner to use our new databunch with all the images.

# +
learn_cln = cnn_learner(db, models.resnet34, metrics=error_rate)

learn_cln.load('stage-2');
# -

ds, idxs = DatasetFormatter().from_toplosses(learn_cln)

# Make sure you're running this notebook in Jupyter Notebook, not Jupyter Lab. That is accessible via [/tree](/tree), not [/lab](/lab). Running the `ImageCleaner` widget in Jupyter Lab is [not currently supported](https://github.com/fastai/fastai/issues/1539).

# Don't run this in google colab or any other instances running jupyter lab.
# If you do run this on Jupyter Lab, you need to restart your runtime and
# runtime state including all local variables will be lost.
ImageCleaner(ds, idxs, path)

#
# If the code above does not show any GUI(contains images and buttons) rendered by widgets but only text output, that may caused by the configuration problem of ipywidgets. Try the solution in this [link](https://github.com/fastai/fastai/issues/1539#issuecomment-505999861) to solve it.
#

# Flag photos for deletion by clicking 'Delete'. Then click 'Next Batch' to delete flagged photos and keep the rest in that row. `ImageCleaner` will show you a new row of images until there are no more to show. In this case, the widget will show you images until there are none left from `top_losses.ImageCleaner(ds, idxs)`

# You can also find duplicates in your dataset and delete them! To do this, you need to run `.from_similars` to get the potential duplicates' ids and then run `ImageCleaner` with `duplicates=True`. The API works in a similar way as with misclassified images: just choose the ones you want to delete and click 'Next Batch' until there are no more images left.

# Make sure to recreate the databunch and `learn_cln` from the `cleaned.csv` file. Otherwise the file would be overwritten from scratch, losing all the results from cleaning the data from toplosses.

ds, idxs = DatasetFormatter().from_similars(learn_cln)

ImageCleaner(ds, idxs, path, duplicates=True)

# Remember to recreate your ImageDataBunch from your `cleaned.csv` to include the changes you made in your data!

# ## Putting your model in production

# First thing first, let's export the content of our `Learner` object for production:

learn.export()

# This will create a file named 'export.pkl' in the directory where we were working that contains everything we need to deploy our model (the model, the weights but also some metadata like the classes or the transforms/normalization used).

# You probably want to use CPU for inference, except at massive scale (and you almost certainly don't need to train in real-time). If you don't have a GPU that happens automatically. You can test your model on CPU like so:

defaults.device = torch.device('cpu')

img = open_image(path/'black'/'00000021.jpg')
img

# We create our `Learner` in production enviromnent like this, just make sure that `path` contains the file 'export.pkl' from before.

learn = load_learner(path)

pred_class,pred_idx,outputs = learn.predict(img)
pred_class.obj

# So you might create a route something like this ([thanks](https://github.com/simonw/cougar-or-not) to Simon Willison for the structure of this code):
#
# ```python
# @app.route("/classify-url", methods=["GET"])
# async def classify_url(request):
#     bytes = await get_bytes(request.query_params["url"])
#     img = open_image(BytesIO(bytes))
#     _,_,losses = learner.predict(img)
#     return JSONResponse({
#         "predictions": sorted(
#             zip(cat_learner.data.classes, map(float, losses)),
#             key=lambda p: p[1],
#             reverse=True
#         )
#     })
# ```
#
# (This example is for the [Starlette](https://www.starlette.io/) web app toolkit.)

# ## Things that can go wrong

# - Most of the time things will train fine with the defaults
# - There's not much you really need to tune (despite what you've heard!)
# - Most likely are
#   - Learning rate
#   - Number of epochs

# ### Learning rate (LR) too high

learn = cnn_learner(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(1, max_lr=0.5)

# ### Learning rate (LR) too low

learn = cnn_learner(data, models.resnet34, metrics=error_rate)

# Previously we had this result:
#
# ```
# Total time: 00:57
# epoch  train_loss  valid_loss  error_rate
# 1      1.030236    0.179226    0.028369    (00:14)
# 2      0.561508    0.055464    0.014184    (00:13)
# 3      0.396103    0.053801    0.014184    (00:13)
# 4      0.316883    0.050197    0.021277    (00:15)
# ```

learn.fit_one_cycle(5, max_lr=1e-5)

learn.recorder.plot_losses()

# As well as taking a really long time, it's getting too many looks at each image, so may overfit.

# ### Too few epochs

learn = cnn_learner(data, models.resnet34, metrics=error_rate, pretrained=False)

learn.fit_one_cycle(1)

# ### Too many epochs

np.random.seed(42)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.9, bs=32, 
        ds_tfms=get_transforms(do_flip=False, max_rotate=0, max_zoom=1, max_lighting=0, max_warp=0
                              ),size=224, num_workers=4).normalize(imagenet_stats)

learn = cnn_learner(data, models.resnet50, metrics=error_rate, ps=0, wd=0)
learn.unfreeze()

learn.fit_one_cycle(40, slice(1e-6,1e-4))
