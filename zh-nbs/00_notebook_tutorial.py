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

# # 00_notebook_tutorial

# **Important note:** You should always work on a duplicate of the course notebook. On the page you used to open this, tick the box next to the name of the notebook and click duplicate to easily create a new version of this notebook.<br>
# You will get errors each time you try to update your course repository if you don't do this, and your changes will end up being erased by the original course version.<br>

# **重要提示:** 你应该在课程notebook的副本上工作。在你打开notebook的页面上，勾选notebook名称旁的选择框，然后点击复制就能轻松创建一个新的notebook副本了。<br>
# 如果你不这样做，那么当你尝试更新课程资源库时就会报错，你的改动会被课程的原始内容所覆盖。

# # Welcome to Jupyter Notebooks!

# # 欢迎来到Jupyter Notebooks！

# If you want to learn how to use this tool you've come to the right place. This article will teach you all you need to know to use Jupyter Notebooks effectively. You only need to go through Section 1 to learn the basics and you can go into Section 2 if you want to further increase your productivity.<br>

# 如果你想学习如何使用这个工具，你来对地方了。这篇文章将教你高效使用jupyter notebook的所有应知应会的内容。你只需完成第1节的内容，就能学到基础知识，如果你想进一步生提高生产率，可以去学习第2节的内容。

# You might be reading this tutorial in a web page (maybe Github or the course's webpage). We strongly suggest to read this tutorial in a (yes, you guessed it) Jupyter Notebook. This way you will be able to actually *try* the different commands we will introduce here.<br>

# 你可能正在通过网页来阅读这篇教程（可能是github网站，或课程的网页上）。我们强烈建议你（没错，你猜对了）在jupyter notebook中来阅读本教程。这种方式可以让你实际*尝试*在本文中介绍到的不同命令。

# ## Section 1: Need to Know 

# ## 第1节：需知

# ### Introduction 简介

# Let's build up from the basics, what is a Jupyter Notebook? Well, you are reading one. It is a document made of cells. You can write like I am writing now (markdown cells) or you can perform calculations in Python (code cells) and run them like this:<br>

# 让我们从最基础的部分开始说起，Jupyter Notebook是什么? 你现在看到的就是一个notebook。它是由一些单元格(cells)组成的文档。你可以像我这样（使用markdown cells）写入内容，你也可以（使用code cells）执行Python中的计算程序并且像下面这样来运行它们：

1+1

# Cool huh? This combination of prose and code makes Jupyter Notebook ideal for experimentation: we can see the rationale for each experiment, the code and the results in one comprehensive document. In fast.ai, each lesson is documented in a notebook and you can later use that notebook to experiment yourself. <br>

# 是不是很cool?这种将普通文本和代码结合起来的模式，使得Jupyter Notebook成为做实验的绝佳选择：在一篇综合性文档中，我们可以既可以看到每个实验的原理讲解，又可以看到对应代码，甚至还有代码运行后的结果。在fast.ai课程里，每一节课的内容都以notebook方式来呈现，随后你也可以自己使用对应的notebook来做实验。

# Other renowned institutions in academy and industry use Jupyter Notebook: Google, Microsoft, IBM, Bloomberg, Berkeley and NASA among others. Even Nobel-winning economists [use Jupyter Notebooks](https://paulromer.net/jupyter-mathematica-and-the-future-of-the-research-paper/)  for their experiments and some suggest that Jupyter Notebooks will be the [new format for research papers](https://www.theatlantic.com/science/archive/2018/04/the-scientific-paper-is-obsolete/556676/).<br>

# 许多在学术界和工业界久负盛名的机构也在使用Jupyter Notebook，比如Google, Microsoft，IBM，Bloomberg，Berkeley以及NASA等，甚至诺贝尔经济学奖得主也在[使用Jupyter Notebooks](https://paulromer.net/jupyter-mathematica-and-the-future-of-the-research-paper/) 来进行实验，其中有一些经济学奖得主认为Jupyter Notebook将成为[新的学术论文格式](https://www.theatlantic.com/science/archive/2018/04/the-scientific-paper-is-obsolete/556676/)。

# ### Writing 写作

# A type of cell in which you can write like this is called _Markdown_. [_Markdown_](https://en.wikipedia.org/wiki/Markdown) is a very popular markup language. To specify that a cell is _Markdown_ you need to click in the drop-down menu in the toolbar and select _Markdown_.<br>

# _Markdown_ 是jupyter notebook里单元格的一种类型，它可以让你进行本文写作。[Markdown](https://en.wikipedia.org/wiki/Markdown) 是一种非常流行的标记语言。为了指定一个单元格为*_Markdown_*,你需要点击工具栏中的下拉菜单并且选择*_Markdown_*。

# Click on the the '+' button on the left and select _Markdown_ from the toolbar.<br>

# 点击左边的“+”按钮，从工具栏中选择*_Markdown_*。

# Now you can type your first _Markdown_ cell. Write 'My first markdown cell' and press run.<br>

# 现在你可以创建你的第一个*_Markdown_*单元格了。在单元格中输入“My first markdown cell”并点击run。

# ![](http://ml.xiniuedu.com/fastai/0/1.png)

# You should see something like this: <br>

# 你将看到下面的内容：

# My first markdown cell

# Now try making your first _Code_ cell: follow the same steps as before but don't change the cell type (when you add a cell its default type is _Code_). Type something like 3/2. You should see '1.5' as output.<br>

# 现在试着创建你的第一个*_Code_*单元格：遵循前面介绍的步骤，但是不要修改单元格的类型（当你添加一个单元格时，它的默认类型就是*_Code_*）。输入一些代码，比如3/2，那么你的输出为“1.5”。

3/2

# ### Modes 模式

# If you made a mistake in your *Markdown* cell and you have already ran it, you will notice that you cannot edit it just by clicking on it. This is because you are in **Command Mode**. Jupyter Notebooks have two distinct modes:<br>

# 如果你在*Markdown*单元格中犯了错误并且已经运行过此单元格，你会发现不能仅通过点击它来进行编辑。这是因为你处于**命令模式**。Jupyter notebooks有两种不同的工作模式：<br>

# 1. **Edit Mode**: Allows you to edit a cell's content.<br>
#    **编辑模式**：允许你对单个单元格的内容进行编辑。

# 2. **Command Mode**: Allows you to edit the notebook as a whole and use keyboard shortcuts but not edit a cell's content. <br>
#    **命令模式**：允许你使用键盘快捷键，将notebook作为一个整体进行编辑,但不能对单个单元格的内容进行编辑。

# You can toggle between these two by either pressing <kbd>ESC</kbd> and <kbd>Enter</kbd> or clicking outside a cell or inside it (you need to 
# double click if its a Markdown cell). You can always know which mode you're on since the current cell has a green border if in **Edit Mode** and a blue border in **Command Mode**. Try it!<br>

# 你可以在这两种模式间转换，方法是同时按下<kbd>ESC</kbd>和<kbd>Enter</kbd>键，或者通过点击一个单元格的外面或者里面来切换（如果是Markdown单元格，你需要双击实现模式切换）。你总是可以通过观察当前单元格的边框颜色来判断当前单元格处于什么模式：如果边框是绿色则表示处在**编辑模式**，如果是蓝色边框则表示处在**命令模式**。试一试吧!

# ### Other Important Considerations 其他重要考虑因素

# 1. Your notebook is autosaved every 120 seconds. If you want to manually save it you can just press the save button on the upper left corner or press <kbd>s</kbd> in **Command Mode**.<br>

# 你的notebook每过120秒就将自动保存。如果你希望手工保存，只需点击左上角的save按钮即可，或者在**命令模式**下按下<kbd>s</kbd>键。

# ![](http://ml.xiniuedu.com/fastai/0/2.png)

# 2. To know if your kernel is computing or not you can check the dot in your upper right corner. If the dot is full, it means that the kernel is working. If not, it is idle. You can place the mouse on it and see the state of the kernel be displayed.<br>

# 如果你想知道你的kernel是否在运行中，你可以检查右上角的圆点。如果是实心的，表示kernel正在工作中，如果是空心的，则表示kernel空闲。你也可以将鼠标悬浮于圆点上，来查看kernel的状态。

# ![](http://ml.xiniuedu.com/fastai/0/3.png)

# 3. There are a couple of shortcuts you must know about which we use **all** the time (always in **Command Mode**). These are:<br>
# （处于**命令模式**时)有一些我们**总是**要用的键盘快捷键，你必须掌握。如下所示：
#
# <kbd>Shift</kbd>+<kbd>Enter</kbd>: Runs the code or markdown on a cell<br>
# <kbd>Shift</kbd>+<kbd>Enter</kbd>：运行一个单元格中的代码或者格式化文本
#
# <kbd>Up Arrow</kbd>+<kbd>Down Arrow</kbd>: Toggle across cells<br>
# <kbd>Up Arrow</kbd>+<kbd>Down Arrow</kbd>：在单元格之间切换选择
#
#
# <kbd>b</kbd>: Create new cell<br>
# <kbd>b</kbd>： 创建一个新的单元格
#
# <kbd>0</kbd>+<kbd>0</kbd>: Reset Kernel<br>
# <kbd>0</kbd>+<kbd>0</kbd>： 重置 Kernel
#
# You can find more shortcuts in the Shortcuts section below.<br>
# 在下面的章节，你还会看到更多快捷键的说明。

# 4. You may need to use a terminal in a Jupyter Notebook environment (for example to git pull on a repository). That is very easy to do, just press 'New' in your Home directory and 'Terminal'. Don't know how to use the Terminal? We made a tutorial for that as well. You can find it [here](https://course.fast.ai/terminal_tutorial.html).<br>

# 你可能需要在Jupyter Notebook的环境中使用terminal（比如通过git pull指令拉取一个repo）。这也非常简单，只需要在你的首页点击“New”，再选择“Terminal”即可。不知道具体怎么用Terminal?我们准备了一篇教程，你可以在 [这里](https://course.fast.ai/terminal_tutorial.html) 找到。

# ![](http://ml.xiniuedu.com/fastai/0/4.png)

# That's it. This is all you need to know to use Jupyter Notebooks. That said, we have more tips and tricks below ↓↓↓<br>

# 好了，这就是使用Jupyter Notebooks时，你需要知道的知识点。当然了，下面还会介绍更多小技巧↓↓↓

# ## Section 2: Going deeper 

# ## 第2节：更进一步

# + [markdown] hide_input=false
# ### Markdown formatting  设定markdown的格式
# -

# #### Italics, Bold, Strikethrough, Inline, Blockquotes and Links 

# #### 斜体，粗体，删除线，内联，引用和链接

# The five most important concepts to format your code appropriately when using markdown are:<br>

# 当你使用markdown时，有五种最重要的格式设定，它们的作用如下：

# 1. *Italics*: Surround your text with '\_' or '\*'  
# *斜体*: 在文本两边包裹上“\_”或者“\*” <br>

# 2. **Bold**: Surround your text with '\__' or '\**'  
# **粗体**: 在文本两边包裹上“\__”或者“**”<br>

# 3. `inline`: Surround your text with '\`'  
# `内联`: 文本两边包裹上“\`”<br>

# 4.  > blockquote: Place '\>' before your text.  
# > 引用：在文本前加上前缀“\>”<br>

# 5.  [Links](https://course.fast.ai/): Surround the text you want to link with '\[\]' and place the link adjacent to the text, surrounded with '()' <br>
# [链接](https://course.fast.ai/)： 在文本两边包裹上 “\[\]”（这里是方括号）,并且紧跟着将链接文本放在“()”中

# #### Headings 标题

# Notice that including a hashtag before the text in a markdown cell makes the text a heading. The number of hashtags you include will determine the priority of the header ('#' is level one, '##' is level two, '###' is level three and '####' is level four). We will add three new cells with the '+' button on the left to see how every level of heading looks.<br>

# 在一个markdown单元格的文本前添加一个“#”,就可将该文本设定为标题了。“#”的个数决定了文本的优先级别。（“#”表示一级标题，“##”表示二级标题，“###”表示三级标题，“####”表示四级标题）。我们通过点击“+”来添加三个新的单元格来演示各个级别的标题都是什么样子的。

# Double click on some headings and find out what level they are!<br>

# 双击下面的标题，看看他们都是什么级别的吧！

# #### Lists 列表

# There are three types of lists in markdown.<br> 

# 在markdown中有三种类型的列表。

# Ordered list: 有序列表
#
# 1. Step 1<br>A.Step 1B
# 2. Step 3

# Unordered list 无序列表
#
# * learning rate 学习速率
# * cycle length 周期长度
# * weight decay 权重衰减

# Task list 任务列表
#
# - [x] Learn Jupyter Notebooks 学习Jupyter Notebooks
#     - [x] Writing 写作
#     - [x] Modes 模式
#     - [x] Other Considerations 其他考虑因素
# - [ ] Change the world 改变世界

# Double click on each to see how they are built!  

# 双击查看这些列表是怎么构建出来的！

# ### Code Capabilities 代码能力

# **Code** cells are different than **Markdown** cells in that they have an output cell. This means that we can *keep* the results of our code within the notebook and share them. Let's say we want to show a graph that explains the result of an experiment. We can just run the necessary cells and save the notebook. The output will be there when we open it again! Try it out by running the next four cells.<br>

# **Code**单元格和**Markdown**单元格是不同类型的单元格，因为**Code**单元格中有一个输出单元格。这意味着我们可以在notebook中 *保留* 代码执行结果，并分享它们。当我们想要展示实验结果的图表时，我们只需要运行必要的单元格并保存notebook。运行结果会在我们再次打开时显示出来！试试看运行接下来的4个单元格吧！

# Import necessary libraries
from fastai.vision import * 
import matplotlib.pyplot as plt

from PIL import Image

a = 1
b = a + 1
c = b + a + 1
d = c + b + a + 1
a, b, c ,d

plt.plot([a,b,c,d])
plt.show()

# We can also print images while experimenting. I am watching you.<br>

# 我们也可以在做实验过程中显示一些图片。（这只猫的图片就像在说）“我在看着你哦”。

Image.open('images/notebook_tutorial/cat_example.jpg')

# ### Running the app locally 本地运行app

# You may be running Jupyter Notebook from an interactive coding environment like Gradient, Sagemaker or Salamander. You can also run a Jupyter Notebook server from your local computer. What's more, if you have installed Anaconda you don't even need to install Jupyter (if not, just `pip install jupyter`).<br>

# 你可能在Gradient, Sagemaker或者Salamander，这样的交互式编码环境中运行Jupyter Notebook。你也可以在本地计算机上运行一个Jupyter Notebook服务器。此外，如果你安装了Anaconda，你甚至不用单独安装Jupyter（如果没有安装的话，只要运行一下`pip install jupyter`就可以了)。

# You just need to run `jupyter notebook` in your terminal. Remember to run it from a folder that contains all the folders/files you will want to access. You will be able to open, view and edit files located within the directory in which you run this command but not files in parent directories.<br>

# 你只需要在你的terminal上运行`jupyter notebook`命令即可。记住在包含你希望访问的文件夹/文件的总文件夹那里来运行这条命令。这样你就可以打开，查看和编辑，运行了`jupyter notebook`命令的文件夹中的文件了，但是记在父目录里面的文件是不能打开查看或者编辑的。<br>

# If a browser tab does not open automatically once you run the command, you should CTRL+CLICK the link starting with 'https://localhost:' and this will open a new tab in your default browser.<br>

# 如果你运行了上面的命令，却没有自动打开浏览器，你也可以按住CTRL键，然后点击以 “https://localhost:” 开头的链接，这样你的默认浏览器中就会打开一个新的标签页。

# ### Creating a notebook 创建一个notebook

# Click on 'New' in the upper left corner and 'Python 3' in the drop-down list (we are going to use a [Python kernel](https://github.com/ipython/ipython) for all our experiments).<br>

# 点击左上角的“New”按钮，随后在下拉列表中选择“Python 3”（我们将在我们的所有实验中使用一个[Python内核](https://github.com/ipython/ipython)）
#

# ![](http://ml.xiniuedu.com/fastai/0/5.png)

# Note: You will sometimes hear people talking about the Notebook 'kernel'. The 'kernel' is just the Python engine that performs the computations for you. <br>

# 注意：你有时可能听到人们谈论Notebook “kernel”，“kernel”就是替你执行计算的Python引擎。

# ### Shortcuts and tricks 快捷键和技巧

# #### Command Mode Shortcuts 命令模式下的快捷键

# There are a couple of useful keyboard shortcuts in `Command Mode` that you can leverage to make Jupyter Notebook faster to use. Remember that to switch back and forth between `Command Mode` and `Edit Mode` with <kbd>Esc</kbd> and <kbd>Enter</kbd>.<br>

# 在`命令模式`下有一些可以提高效率的快捷键。记住在`命令模式`和`编辑`模式间来回切换的快捷键是<kbd>Esc</kbd> 和 <kbd>Enter</kbd>。<br><br>

# <kbd>m</kbd>: Convert cell to Markdown 将单元格转换为Markdown单元格

# <kbd>y</kbd>: Convert cell to Code 将单元格转换为Code代码单元格

# <kbd>D</kbd>+<kbd>D</kbd>: Delete cell 删除单元格

# <kbd>o</kbd>: Toggle between hide or show output 切换显示或者隐藏输出信息

# <kbd>Shift</kbd>+<kbd>Arrow up上箭头/Arrow down下箭头</kbd>: Selects multiple cells. Once you have selected them you can operate on them like a batch (run, copy, paste etc).
# 用于选择多个单元格。一旦你选中了多个单元格，你就可以批量操作他们（比如运行，复制，粘贴等操作）。

# <kbd>Shift</kbd>+<kbd>M</kbd>: Merge selected cells. 合并选中的单元格为一个单元格

# <kbd>Shift</kbd>+<kbd>Tab</kbd>: [press once] Tells you which parameters to pass on a function 
# [按键一次]提示函数有哪些参数

# <kbd>Shift</kbd>+<kbd>Tab</kbd>: [press three times] Gives additional information on the method 
# [按键三次] 提示这个方法的更多信息

# #### Cell Tricks 单元格小技巧

from fastai import*
from fastai.vision import *

# There are also some tricks that you can code into a cell.<br> 

# 这里还有一些在单元格编码的一些小技巧。

# `?function-name`: Shows the definition and docstring for that function <br>

# `?function-name`：显示该函数的定义和文档信息

# ?ImageDataBunch

# `??function-name`: Shows the source code for that function<br>

# `??function-name`：显示函数的源代码

??ImageDataBunch

# `doc(function-name)`: Shows the definition, docstring **and links to the documentation** of the function
# (only works with fastai library imported)<br>

# `doc(function-name)`：显示定义、文档信息以及**详细文档的链接**（只有在import导入了fastai库之后才能工作）

doc(ImageDataBunch)

# #### Line Magics 

# Line magics are functions that you can run on cells and take as an argument the rest of the line from where they are called. You call them by placing a '%' sign before the command. The most useful ones are:<br>

# Line magics是可以在单元格中运行并且将该行的其他信息作为参数的函数。通过在命令之前添加一个“%”来调用他们。最有用的是以下几个：

# `%matplotlib inline`: This command ensures that all matplotlib plots will be plotted in the output cell within the notebook and will be kept in the notebook when saved.<br>

# `%matplotlib inline`：该命令确保所有的matplotlib图表都将绘制在notebook的输出单元格中，并且在保存时一并保留在notebook中。

# `%reload_ext autoreload`, `%autoreload 2`: Reload all modules before executing a new line. If a module is edited, it is not necessary to rerun the import commands, the modules will be reloaded automatically.<br>

# `%reload_ext autoreload`,`%autoreload 2`：这两条命令指示在执行新的行代码时重新加载所有的模块。如果一个模块修改过了，没有必要再次运行import命令，模块将自动重新加载。

# These three commands are always called together at the beginning of every notebook. <br>

# 通常这3条命令在每一个notebook的起始部分被一起调用。

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

# `%timeit`: Runs a line a ten thousand times and displays the average time it took to run it.<br>

# `%timeit`：这个命令将会运行一行代码1000次并且显示平均运行的时间。

# %timeit [i+1 for i in range(1000)]

# `%debug`: Allows to inspect a function which is showing an error using the [Python debugger](https://docs.python.org/3/library/pdb.html).<br>

# `%debug`：允许你使用[Python调试器](https://docs.python.org/3/library/pdb.html)来检查报错的函数。

for i in range(1000):
    a = i+1
    b = 'string'
    c = b+1

# %debug
