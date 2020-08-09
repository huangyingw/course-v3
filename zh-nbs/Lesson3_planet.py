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

# # Lesson3_planet

# ## Multi-label prediction with Planet Amazon dataset
# ## 基于Planet Amazon数据集的多标签分类预测

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

from fastai.vision import *

# ## Getting the data 数据获取

# The planet dataset isn't available on the [fastai dataset page](https://course.fast.ai/datasets) due to copyright restrictions. 
#
# 由于版权原因，我们不能将亚马逊雨林的数据集放在fastai网页的数据集页面。
#
# You can download it from Kaggle however. Let's see how to do this by using the [Kaggle API](https://github.com/Kaggle/kaggle-api) as it's going to be pretty useful to you if you want to join a competition or use other Kaggle datasets later on.
#
# 不过，你可以通过Kaggle网站来进行下载。让我们来看一下怎样通过 Kaggle API 下载数据。这项技能很重要，因为未来你可能会需要使用这个API来下载竞赛数据，或者使用其他Kaggle的数据集。
#
# First, install the Kaggle API by uncommenting the following line and executing it, or by executing it in your terminal (depending on your platform you may need to modify this slightly to either add `source activate fastai` or similar, or prefix `pip` with a path. Have a look at how `conda install` is called for your platform in the appropriate *Returning to work* section of https://course.fast.ai/. (Depending on your environment, you may also need to append "--user" to the command.)
#
# 首先，要安装Kaggle API需要取消下一行的注释并运行代码，或者在terminal里执行（这取决于你的运行系统，你可能需要对这个代码进行一点修改。你可能会需要添加`source activate fastai`来略作修改，也可以在下面的代码之前加入`pip`, 还可以在代码之后加上`--user`。 在你的系统中究竟该怎样使用`conda install`，可以参照https://course.fast.ai/ 页面的*Returning to work* 部分）。

# +
# ! {sys.executable} -m pip install kaggle --upgrade
# -

# Then you need to upload your credentials from Kaggle on your instance. Login to kaggle and click on your profile picture on the top left corner, then 'My account'. Scroll down until you find a button named 'Create New API Token' and click on it. This will trigger the download of a file named 'kaggle.json'.
#
# 接下来你需要在你的代码中上传你的身份验证资料。你需要登入Kaggle，在左上角点击你的头像，选择`我的账户`，然后向下滑动，直到你找到`创建新API许可权`，并点击这个按钮。这样会产生一个自动下载的名为`kaggle.json`的文件。
#
# Upload this file to the directory this notebook is running in, by clicking "Upload" on your main Jupyter page, then uncomment and execute the next two commands (or run them in a terminal). For Windows, uncomment the last two commands.
#
# 在Jupyter的主页上点击`Upload`，将下载好的文件上传到这个notebook运行的路径中。然后运行下面两行代码，你也可以直接在terminal里运行。如果你是windows用户，只运行后面两行代码即可。

# +
# # ! mkdir -p ~/.kaggle/
# # ! mv kaggle.json ~/.kaggle/

# For Windows, uncomment these two commands
# # ! mkdir %userprofile%\.kaggle
# # ! move kaggle.json %userprofile%\.kaggle
# -

# You're all set to download the data from [planet competition](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space). You **first need to go to its main page and accept its rules**, and run the two cells below (uncomment the shell commands to download and unzip the data). If you get a `403 forbidden` error it means you haven't accepted the competition rules yet (you have to go to the competition page, click on *Rules* tab, and then scroll to the bottom to find the *accept* button).
#
# 现在你可以开始从[planet competition](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space)下载数据。**需要注意的是，你要先去kaggle主页上接受相应的条款**，之后运行下面的两行代码，取消注释并且解压数据。如果你看到`403 forbidden`的字样，这代表你还没有接受比赛的条款。你需要去这个比赛的主页，点击*Rules* 按钮，下拉到页面最下方并点击*accept* 按钮。

path = Config.data_path()/'planet'
path.mkdir(parents=True, exist_ok=True)
path

# +
# # ! kaggle competitions download -c planet-understanding-the-amazon-from-space -f train-jpg.tar.7z -p {path}  
# # ! kaggle competitions download -c planet-understanding-the-amazon-from-space -f train_v2.csv -p {path}  
# # ! unzip -q -n {path}/train_v2.csv.zip -d {path}
# -

# To extract the content of this file, we'll need 7zip, so uncomment the following line if you need to install it (or run `sudo apt install p7zip-full` in your terminal).
#
# 我们需要7zip来提取所有文件，因此如果你需要安装7zip, 你可以取消下面这一行代码的注释，然后运行就可以安装了（或者如果你是苹果系统用户，可以在terminal里运行`sudo apt install p7zip-full`）。

# +
# # ! conda install --yes --prefix {sys.prefix} -c haasad eidl7zip
# -

# And now we can unpack the data (uncomment to run - this might take a few minutes to complete).
#
# 我们可以运行下面的代码来解压数据（取消注释再运行——这可能需要几分钟才能完成）。

# +
# ! 7za -bd -y -so x {path}/train-jpg.tar.7z | tar xf - -C {path.as_posix()}
# -

# ## Multiclassification多标签分类问题

# Contrary to the pets dataset studied in last lesson, here each picture can have multiple labels. If we take a look at the csv file containing the labels (in 'train_v2.csv' here) we see that each 'image_name' is associated to several tags separated by spaces.
#
# 与上节课学习的宠物数据集相比，这节课的数据集里每张图片都有多个标签。如果我们看一下导入的csv数据（在“train_v2.csv”这里），就可以看见每个图片名都有好几个由空格分开的标签。

df = pd.read_csv(path/'train_v2.csv')
df.head()

# To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html), we then need to using `ImageList` (and not `ImageDataBunch`). This will make sure the model created has the proper loss function to deal with the multiple classes.
#
# 我们将这些数据和标签用 [data block API](https://docs.fast.ai/data_block.html) 转化成`DataBunch`，接着需要使用`ImageList` （而不是`ImageDataBunch`）。这样做可以保证模型有正确的损失函数来处理多标签的问题。

tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)

# We use parentheses around the data block pipeline below, so that we can use a multiline statement without needing to add '\\'.
#
# 我们在下面的代码前后使用括号，这样可以很方便的写入多行代码而不需要给每行末尾加“\\”。

np.random.seed(42)
src = (ImageList.from_csv(path, 'train_v2.csv', folder='train-jpg', suffix='.jpg')
       .split_by_rand_pct(0.2)
       .label_from_df(label_delim=' '))

data = (src.transform(tfms, size=128)
        .databunch().normalize(imagenet_stats))

# `show_batch` still works, and show us the different labels separated by `;`.

data.show_batch(rows=3, figsize=(12,9))

# To create a `Learner` we use the same function as in lesson 1. Our base architecture is resnet50 again, but the metrics are a little bit differeent: we use `accuracy_thresh` instead of `accuracy`. In lesson 1, we determined the predicition for a given class by picking the final activation that was the biggest, but here, each activation can be 0. or 1. `accuracy_thresh` selects the ones that are above a certain threshold (0.5 by default) and compares them to the ground truth.
#
# 我们用第一课里同样的函数来创建一个`Learner`。我们的基础架构依然是resnet50, 但这次使用的度量函数有点不同：我们会使用`accuracy_thresh`来代替 `accuracy`。在第一课里，我们采用的分组标签是给定品种的最终激活函数的最大值，但是在这里，每个激活函数的值可以是0或1，由`accuracy_thresh`选取所有高于某个“阈值”（默认为0.5）的图像，然后与真实的标签做对比。
#
# As for Fbeta, it's the metric that was used by Kaggle on this competition. See [here](https://en.wikipedia.org/wiki/F1_score) for more details.
#
# 至于Fbeta, 它是这项Kaggle比赛使用的测度。欲知详情，可以看[这里](https://en.wikipedia.org/wiki/F1_score)

arch = models.resnet50

acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner(data, arch, metrics=[acc_02, f_score])

# We use the LR Finder to pick a good learning rate.
#
# 我们使用LR Finder来选取一个好的学习率。

learn.lr_find()

learn.recorder.plot()

# Then we can fit the head of our network.
#
# 接下来，我们可以用找好的学习率来训练神经网络模型了。

learn.fit_one_cycle(5, slice(lr))

learn.save('stage-1-rn50')

# ...And fine-tune the whole model:
#
# 接下来给整个模型调参：

learn.unfreeze()

learn.lr_find()
learn.recorder.plot()

learn.fit_one_cycle(5, slice(1e-5, lr/5))

learn.save('stage-2-rn50')

# +
data = (src.transform(tfms, size=256)
        .databunch().normalize(imagenet_stats))

learn.data = data
data.train_ds[0][0].shape
# -

learn.freeze()

learn.lr_find()
learn.recorder.plot()

lr=1e-2/2

learn.fit_one_cycle(5, slice(lr))

learn.save('stage-1-256-rn50')

learn.unfreeze()

learn.fit_one_cycle(5, slice(1e-5, lr/5))

learn.recorder.plot_losses()

learn.save('stage-2-256-rn50')

# You won't really know how you're going until you submit to Kaggle, since the leaderboard isn't using the same subset as we have for training. But as a guide, 50th place (out of 938 teams) on the private leaderboard was a score of `0.930`.
#
# 正如我们训练时做的那样，排名榜单上使用了不同的数据子集, 如果你没有在Kaggle提交你的模型，你就无法知道自己的模型表现得怎么样。不过作为一个参考，`0.930`在非公开的榜单上大约是在938个团队里排到第50名。

learn.export()

# ## fin

# (This section will be covered in part 2 - please don't ask about it just yet! :) )
#
# （这个部分在part2里已经被覆盖——就别在意这些细节啦 :)）

# +
# #! kaggle competitions download -c planet-understanding-the-amazon-from-space -f test-jpg.tar.7z -p {path}  
#! 7za -bd -y -so x {path}/test-jpg.tar.7z | tar xf - -C {path}
# #! kaggle competitions download -c planet-understanding-the-amazon-from-space -f test-jpg-additional.tar.7z -p {path}  
#! 7za -bd -y -so x {path}/test-jpg-additional.tar.7z | tar xf - -C {path}
# -

test = ImageList.from_folder(path/'test-jpg').add(ImageList.from_folder(path/'test-jpg-additional'))
len(test)

learn = load_learner(path, test=test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)

thresh = 0.2
labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]

labelled_preds[:5]

fnames = [f.name[:-4] for f in learn.data.test_ds.items]

df = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])

df.to_csv(path/'submission.csv', index=False)

# ! kaggle competitions submit planet-understanding-the-amazon-from-space -f {path/'submission.csv'} -m "My submission"

# Private Leaderboard score: 0.9296 (around 80th)
#
# 内部排名榜单得分：0.9296（约第80位）
