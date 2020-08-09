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

#export
TEST = 'test'

# ## Export

# !python notebook2script.py 00_exports.ipynb

# ### How it works:

import json
d = json.load(open('00_exports.ipynb','r'))['cells']

d[0]


