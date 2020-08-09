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

# **Important note:** You should always work on a duplicate of the course notebook. On the page you used to open this, tick the box next to the name of the notebook and click duplicate to easily create a new version of this notebook.
#
# You will get errors each time you try to update your course repository if you don't do this, and your changes will end up being erased by the original course version.

# # Welcome to Jupyter Notebooks!

# If you want to learn how to use this tool you've come to the right place. This article will teach you all you need to know to use Jupyter Notebooks effectively. You only need to go through Section 1 to learn the basics and you can go into Section 2 if you want to further increase your productivity.

# You might be reading this tutorial in a web page (maybe Github or the course's webpage). We strongly suggest to read this tutorial in a (yes, you guessed it) Jupyter Notebook. This way you will be able to actually *try* the different commands we will introduce here.

# ## Section 1: Need to Know

# ### Introduction

# Let's build up from the basics, what is a Jupyter Notebook? Well, you are reading one. It is a document made of cells. You can write like I am writing now (markdown cells) or you can perform calculations in Python (code cells) and run them like this:

1+1

# Cool huh? This combination of prose and code makes Jupyter Notebook ideal for experimentation: we can see the rationale for each experiment, the code and the results in one comprehensive document. In fast.ai, each lesson is documented in a notebook and you can later use that notebook to experiment yourself. 
#
# Other renowned institutions in academy and industry use Jupyter Notebook: Google, Microsoft, IBM, Bloomberg, Berkeley and NASA among others. Even Nobel-winning economists [use Jupyter Notebooks](https://paulromer.net/jupyter-mathematica-and-the-future-of-the-research-paper/)  for their experiments and some suggest that Jupyter Notebooks will be the [new format for research papers](https://www.theatlantic.com/science/archive/2018/04/the-scientific-paper-is-obsolete/556676/).

# ### Writing

# A type of cell in which you can write like this is called _Markdown_. [_Markdown_](https://en.wikipedia.org/wiki/Markdown) is a very popular markup language. To specify that a cell is _Markdown_ you need to click in the drop-down menu in the toolbar and select _Markdown_.

# Click on the '+' button on the left and select _Markdown_ from the toolbar.

# Now you can type your first _Markdown_ cell. Write 'My first markdown cell' and press run.

# ![add](images/notebook_tutorial/add.png)

# You should see something like this:

# My first markdown cell

# Now try making your first _Code_ cell: follow the same steps as before but don't change the cell type (when you add a cell its default type is _Code_). Type something like 3/2. You should see '1.5' as output.

3/2

# ### Modes

# If you made a mistake in your *Markdown* cell and you have already ran it, you will notice that you cannot edit it just by clicking on it. This is because you are in **Command Mode**. Jupyter Notebooks have two distinct modes:
#
# 1. **Edit Mode**: Allows you to edit a cell's content.
#
# 2. **Command Mode**: Allows you to edit the notebook as a whole and use keyboard shortcuts but not edit a cell's content. 
#
# You can toggle between these two by either pressing <kbd>ESC</kbd> and <kbd>Enter</kbd> or clicking outside a cell or inside it (you need to double click if its a Markdown cell). You can always know which mode you're on since the current cell has a green border if in **Edit Mode** and a blue border in **Command Mode**. Try it!

# ### Other Important Considerations

# 1. Your notebook is autosaved every 120 seconds. If you want to manually save it you can just press the save button on the upper left corner or press <kbd>s</kbd> in **Command Mode**.

# ![Save](images/notebook_tutorial/save.png)

# 2. To know if your kernel is computing or not you can check the dot in your upper right corner. If the dot is full, it means that the kernel is working. If not, it is idle. You can place the mouse on it and see the state of the kernel be displayed.

# ![Busy](images/notebook_tutorial/busy.png)

# 3. There are a couple of shortcuts you must know about which we use **all** the time (always in **Command Mode**). These are:
#
# <kbd>Shift</kbd>+<kbd>Enter</kbd>: Runs the code or markdown on a cell
#
# <kbd>Up Arrow</kbd>+<kbd>Down Arrow</kbd>: Toggle across cells
#
# <kbd>b</kbd>: Create new cell
#
# <kbd>0</kbd>+<kbd>0</kbd>: Reset Kernel
#
# You can find more shortcuts in the Shortcuts section below.

# 4. You may need to use a terminal in a Jupyter Notebook environment (for example to git pull on a repository). That is very easy to do, just press 'New' in your Home directory and 'Terminal'. Don't know how to use the Terminal? We made a tutorial for that as well. You can find it [here](https://course.fast.ai/terminal_tutorial.html).

# ![Terminal](images/notebook_tutorial/terminal.png)

# That's it. This is all you need to know to use Jupyter Notebooks. That said, we have more tips and tricks below ↓↓↓

# ## Section 2: Going deeper

# + [markdown] hide_input=false
# ### Markdown formatting
# -

# #### Italics, Bold, Strikethrough, Inline, Blockquotes and Links

# The five most important concepts to format your code appropriately when using markdown are:
#     
# 1. *Italics*: Surround your text with '\_' or '\*'
# 2. **Bold**: Surround your text with '\__' or '\**'
# 3. `inline`: Surround your text with '\`'
# 4.  > blockquote: Place '\>' before your text.
# 5.  [Links](https://course.fast.ai/): Surround the text you want to link with '\[\]' and place the link adjacent to the text, surrounded with '()'
#

# #### Headings

# Notice that including a hashtag before the text in a markdown cell makes the text a heading. The number of hashtags you include will determine the priority of the header ('#' is level one, '##' is level two, '###' is level three and '####' is level four). We will add three new cells with the '+' button on the left to see how every level of heading looks.

# Double click on some headings and find out what level they are!

# #### Lists

# There are three types of lists in markdown.

# Ordered list:
#
# 1. Step 1
#     2. Step 1B
# 3. Step 3

# Unordered list
#
# * learning rate
# * cycle length
# * weight decay

# Task list
#
# - [x] Learn Jupyter Notebooks
#     - [x] Writing
#     - [x] Modes
#     - [x] Other Considerations
# - [ ] Change the world

# Double click on each to see how they are built! 

# ### Code Capabilities

# **Code** cells are different than **Markdown** cells in that they have an output cell. This means that we can _keep_ the results of our code within the notebook and share them. Let's say we want to show a graph that explains the result of an experiment. We can just run the necessary cells and save the notebook. The output will be there when we open it again! Try it out by running the next four cells.

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

# We can also print images while experimenting. I am watching you.

Image.open('images/notebook_tutorial/cat_example.jpg')

# ### Running the app locally

# You may be running Jupyter Notebook from an interactive coding environment like Gradient, Sagemaker or Salamander. You can also run a Jupyter Notebook server from your local computer. What's more, if you have installed Anaconda you don't even need to install Jupyter (if not, just `pip install jupyter`).
#
# You just need to run `jupyter notebook` in your terminal. Remember to run it from a folder that contains all the folders/files you will want to access. You will be able to open, view and edit files located within the directory in which you run this command but not files in parent directories.
#
# If a browser tab does not open automatically once you run the command, you should CTRL+CLICK the link starting with 'https://localhost:' and this will open a new tab in your default browser.

# ### Creating a notebook

# Click on 'New' in the upper right corner and 'Python 3' in the drop-down list (we are going to use a [Python kernel](https://github.com/ipython/ipython) for all our experiments).
#
# ![new_notebook](images/notebook_tutorial/new_notebook.png)
#
# Note: You will sometimes hear people talking about the Notebook 'kernel'. The 'kernel' is just the Python engine that performs the computations for you. 

# ### Shortcuts and tricks

# #### Command Mode Shortcuts

# There are a couple of useful keyboard shortcuts in `Command Mode` that you can leverage to make Jupyter Notebook faster to use. Remember that to switch back and forth between `Command Mode` and `Edit Mode` with <kbd>Esc</kbd> and <kbd>Enter</kbd>.

# <kbd>m</kbd>: Convert cell to Markdown

# <kbd>y</kbd>: Convert cell to Code

# <kbd>D</kbd>+<kbd>D</kbd>: Delete the cell(if it's not the only cell) or delete the content of the cell and reset cell to Code(if only one cell left)

# <kbd>o</kbd>: Toggle between hide or show output

# <kbd>Shift</kbd>+<kbd>Arrow up/Arrow down</kbd>: Selects multiple cells. Once you have selected them you can operate on them like a batch (run, copy, paste etc).

# <kbd>Shift</kbd>+<kbd>M</kbd>: Merge selected cells.

# <kbd>Shift</kbd>+<kbd>Tab</kbd>: [press these two buttons at the same time, once] Tells you which parameters to pass on a function
#
# <kbd>Shift</kbd>+<kbd>Tab</kbd>: [press these two buttons at the same time, three times] Gives additional information on the method

# #### Cell Tricks

from fastai import *
from fastai.vision import *

# There are also some tricks that you can code into a cell.

# `?function-name`: Shows the definition and docstring for that function

# ?ImageDataBunch

# `??function-name`: Shows the source code for that function

??ImageDataBunch

# `doc(function-name)`: Shows the definition, docstring **and links to the documentation** of the function
# (only works with fastai library imported)

doc(ImageDataBunch)

# #### Line Magics

# Line magics are functions that you can run on cells and take as an argument the rest of the line from where they are called. You call them by placing a '%' sign before the command. The most useful ones are:

# `%matplotlib inline`: This command ensures that all matplotlib plots will be plotted in the output cell within the notebook and will be kept in the notebook when saved.

# `%reload_ext autoreload`, `%autoreload 2`: Reload all modules before executing a new line. If a module is edited, it is not necessary to rerun the import commands, the modules will be reloaded automatically.

# These three commands are always called together at the beginning of every notebook.

# %matplotlib inline
# %reload_ext autoreload
# %autoreload 2

# `%timeit`: Runs a line ten thousand times and displays the average time it took to run it.

# %timeit [i+1 for i in range(1000)]

# `%debug`: Allows to inspect a function which is showing an error using the [Python debugger](https://docs.python.org/3/library/pdb.html).

for i in range(1000):
    a = i+1
    b = 'string'
    c = b+1

# %debug


