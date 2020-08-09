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

# # Pretraining on WT103

# +
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline
# -

#export
from exp.nb_12a import *

# ## Data

# One time download

# [Jump_to lesson 12 video](https://course.fast.ai/videos/?lesson=12&t=7410)

# +
#path = datasets.Config().data_path()
#version = '103' #2

# +
# #! wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-{version}-v1.zip -P {path}
# #! unzip -q -n {path}/wikitext-{version}-v1.zip  -d {path}
# #! mv {path}/wikitext-{version}/wiki.train.tokens {path}/wikitext-{version}/train.txt
# #! mv {path}/wikitext-{version}/wiki.valid.tokens {path}/wikitext-{version}/valid.txt
# #! mv {path}/wikitext-{version}/wiki.test.tokens {path}/wikitext-{version}/test.txt
# -

# Split the articles: WT103 is given as one big text file and we need to chunk it in different articles if we want to be able to shuffle them at the beginning of each epoch.

path = datasets.Config().data_path()/'wikitext-103'


def istitle(line):
    return len(re.findall(r'^ = [^=]* = $', line)) != 0


def read_wiki(filename):
    articles = []
    with open(filename, encoding='utf8') as f:
        lines = f.readlines()
    current_article = ''
    for i,line in enumerate(lines):
        current_article += line
        if i < len(lines)-2 and lines[i+1] == ' \n' and istitle(lines[i+2]):
            current_article = current_article.replace('<unk>', UNK)
            articles.append(current_article)
            current_article = ''
    current_article = current_article.replace('<unk>', UNK)
    articles.append(current_article)
    return articles


train = TextList(read_wiki(path/'train.txt'), path=path) #+read_file(path/'test.txt')
valid = TextList(read_wiki(path/'valid.txt'), path=path)

len(train), len(valid)

sd = SplitData(train, valid)

proc_tok,proc_num = TokenizeProcessor(),NumericalizeProcessor()

ll = label_by_func(sd, lambda x: 0, proc_x = [proc_tok,proc_num])

pickle.dump(ll, open(path/'ld.pkl', 'wb'))

ll = pickle.load( open(path/'ld.pkl', 'rb'))

bs,bptt = 128,70
data = lm_databunchify(ll, bs, bptt)

vocab = ll.train.proc_x[-1].vocab
len(vocab)

# ## Model

dps = np.array([0.1, 0.15, 0.25, 0.02, 0.2]) * 0.2
tok_pad = vocab.index(PAD)

emb_sz, nh, nl = 300, 300, 2
model = get_language_model(len(vocab), emb_sz, nh, nl, tok_pad, *dps)

cbs = [partial(AvgStatsCallback,accuracy_flat),
       CudaCallback, Recorder,
       partial(GradientClipping, clip=0.1),
       partial(RNNTrainer, α=2., β=1.),
       ProgressCallback]

learn = Learner(model, data, cross_entropy_flat, lr=5e-3, cb_funcs=cbs, opt_func=adam_opt())

lr = 5e-3
sched_lr  = combine_scheds([0.3,0.7], cos_1cycle_anneal(lr/10., lr, lr/1e5))
sched_mom = combine_scheds([0.3,0.7], cos_1cycle_anneal(0.8, 0.7, 0.8))
cbsched = [ParamScheduler('lr', sched_lr), ParamScheduler('mom', sched_mom)]

learn.fit(10, cbs=cbsched)

torch.save(learn.model.state_dict(), path/'pretrained.pth')
pickle.dump(vocab, open(path/'vocab.pkl', 'wb'))


