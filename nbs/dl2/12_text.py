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

# # Preprocess text

# +
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline
# -

#export
from exp.nb_11a import *

# ## Data

# We will use the IMDB dataset that consists of 50,000 labeled reviews of movies (positive or negative) and 50,000 unlabelled ones.

# [Jump_to lesson 12 video](https://course.fast.ai/videos/?lesson=12&t=4964)

path = datasets.untar_data(datasets.URLs.IMDB)

path.ls()


# We define a subclass of `ItemList` that will read the texts in the corresponding filenames.

# +
#export
def read_file(fn): 
    with open(fn, 'r', encoding = 'utf8') as f: return f.read()
    
class TextList(ItemList):
    @classmethod
    def from_files(cls, path, extensions='.txt', recurse=True, include=None, **kwargs):
        return cls(get_files(path, extensions, recurse=recurse, include=include), path, **kwargs)
    
    def get(self, i):
        if isinstance(i, Path): return read_file(i)
        return i


# -

# Just in case there are some text log files, we restrict the ones we take to the training, test, and unsupervised folders.

il = TextList.from_files(path, include=['train', 'test', 'unsup'])

# We should expect a total of 100,000 texts.

len(il.items)

# Here is the first one as an example.

txt = il[0]
txt

# For text classification, we will split by the grand parent folder as before, but for language modeling, we take all the texts and just put 10% aside.

sd = SplitData.split_by_func(il, partial(random_splitter, p_valid=0.1))

sd

# ## Tokenizing

# We need to tokenize the dataset first, which is splitting a sentence in individual tokens. Those tokens are the basic words or punctuation signs with a few tweaks: don't for instance is split between do and n't. We will use a processor for this, in conjunction with the [spacy library](https://spacy.io/).

# [Jump_to lesson 12 video](https://course.fast.ai/videos/?lesson=12&t=5070)

#export
import spacy,html

# Before even tokenizeing, we will apply a bit of preprocessing on the texts to clean them up (we saw the one up there had some HTML code). These rules are applied before we split the sentences in tokens.

# +
#export
#special tokens
UNK, PAD, BOS, EOS, TK_REP, TK_WREP, TK_UP, TK_MAJ = "xxunk xxpad xxbos xxeos xxrep xxwrep xxup xxmaj".split()

def sub_br(t):
    "Replaces the <br /> by \n"
    re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
    return re_br.sub("\n", t)

def spec_add_spaces(t):
    "Add spaces around / and #"
    return re.sub(r'([/#])', r' \1 ', t)

def rm_useless_spaces(t):
    "Remove multiple spaces"
    return re.sub(' {2,}', ' ', t)

def replace_rep(t):
    "Replace repetitions at the character level: cccc -> TK_REP 4 c"
    def _replace_rep(m:Collection[str]) -> str:
        c,cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '
    re_rep = re.compile(r'(\S)(\1{3,})')
    return re_rep.sub(_replace_rep, t)
    
def replace_wrep(t):
    "Replace word repetitions: word word word -> TK_WREP 3 word"
    def _replace_wrep(m:Collection[str]) -> str:
        c,cc = m.groups()
        return f' {TK_WREP} {len(cc.split())+1} {c} '
    re_wrep = re.compile(r'(\b\w+\W+)(\1{3,})')
    return re_wrep.sub(_replace_wrep, t)

def fixup_text(x):
    "Various messy things we've seen in documents"
    re1 = re.compile(r'  +')
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>',UNK).replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))
    
default_pre_rules = [fixup_text, replace_rep, replace_wrep, spec_add_spaces, rm_useless_spaces, sub_br]
default_spec_tok = [UNK, PAD, BOS, EOS, TK_REP, TK_WREP, TK_UP, TK_MAJ]
# -

replace_rep('cccc')

replace_wrep('word word word word word ')


# These rules are applies after the tokenization on the list of tokens.

# +
#export
def replace_all_caps(x):
    "Replace tokens in ALL CAPS by their lower version and add `TK_UP` before."
    res = []
    for t in x:
        if t.isupper() and len(t) > 1: res.append(TK_UP); res.append(t.lower())
        else: res.append(t)
    return res

def deal_caps(x):
    "Replace all Capitalized tokens in by their lower version and add `TK_MAJ` before."
    res = []
    for t in x:
        if t == '': continue
        if t[0].isupper() and len(t) > 1 and t[1:].islower(): res.append(TK_MAJ)
        res.append(t.lower())
    return res

def add_eos_bos(x): return [BOS] + x + [EOS]

default_post_rules = [deal_caps, replace_all_caps, add_eos_bos]
# -

replace_all_caps(['I', 'AM', 'SHOUTING'])

deal_caps(['My', 'name', 'is', 'Jeremy'])

# Since tokenizing and applying those rules takes a bit of time, we'll parallelize it using `ProcessPoolExecutor` to go faster.

# +
#export
from spacy.symbols import ORTH
from concurrent.futures import ProcessPoolExecutor

def parallel(func, arr, max_workers=4):
    if max_workers<2: results = list(progress_bar(map(func, enumerate(arr)), total=len(arr)))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(progress_bar(ex.map(func, enumerate(arr)), total=len(arr)))
    if any([o is not None for o in results]): return results


# -

#export
class TokenizeProcessor(Processor):
    def __init__(self, lang="en", chunksize=2000, pre_rules=None, post_rules=None, max_workers=4): 
        self.chunksize,self.max_workers = chunksize,max_workers
        self.tokenizer = spacy.blank(lang).tokenizer
        for w in default_spec_tok:
            self.tokenizer.add_special_case(w, [{ORTH: w}])
        self.pre_rules  = default_pre_rules  if pre_rules  is None else pre_rules
        self.post_rules = default_post_rules if post_rules is None else post_rules

    def proc_chunk(self, args):
        i,chunk = args
        chunk = [compose(t, self.pre_rules) for t in chunk]
        docs = [[d.text for d in doc] for doc in self.tokenizer.pipe(chunk)]
        docs = [compose(t, self.post_rules) for t in docs]
        return docs

    def __call__(self, items): 
        toks = []
        if isinstance(items[0], Path): items = [read_file(i) for i in items]
        chunks = [items[i: i+self.chunksize] for i in (range(0, len(items), self.chunksize))]
        toks = parallel(self.proc_chunk, chunks, max_workers=self.max_workers)
        return sum(toks, [])
    
    def proc1(self, item): return self.proc_chunk([item])[0]
    
    def deprocess(self, toks): return [self.deproc1(tok) for tok in toks]
    def deproc1(self, tok):    return " ".join(tok)


tp = TokenizeProcessor()

txt[:250]

' • '.join(tp(il[:100])[0])[:400]

# ## Numericalizing

# Once we have tokenized our texts, we replace each token by an individual number, this is called numericalizing. Again, we do this with a processor (not so different from the `CategoryProcessor`).

# [Jump_to lesson 12 video](https://course.fast.ai/videos/?lesson=12&t=5491)

# +
#export
import collections

class NumericalizeProcessor(Processor):
    def __init__(self, vocab=None, max_vocab=60000, min_freq=2): 
        self.vocab,self.max_vocab,self.min_freq = vocab,max_vocab,min_freq
    
    def __call__(self, items):
        #The vocab is defined on the first use.
        if self.vocab is None:
            freq = Counter(p for o in items for p in o)
            self.vocab = [o for o,c in freq.most_common(self.max_vocab) if c >= self.min_freq]
            for o in reversed(default_spec_tok):
                if o in self.vocab: self.vocab.remove(o)
                self.vocab.insert(0, o)
        if getattr(self, 'otoi', None) is None:
            self.otoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.vocab)}) 
        return [self.proc1(o) for o in items]
    def proc1(self, item):  return [self.otoi[o] for o in item]
    
    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deproc1(idx) for idx in idxs]
    def deproc1(self, idx): return [self.vocab[i] for i in idx]


# -

# When we do language modeling, we will infer the labels from the text during training, so there's no need to label. The training loop expects labels however, so we need to add dummy ones.

proc_tok,proc_num = TokenizeProcessor(max_workers=8),NumericalizeProcessor()

# %time ll = label_by_func(sd, lambda x: 0, proc_x = [proc_tok,proc_num])

# Once the items have been processed they will become list of numbers, we can still access the underlying raw data in `x_obj` (or `y_obj` for the targets, but we don't have any here).

ll.train.x_obj(0)

# Since the preprocessing tajes time, we save the intermediate result using pickle. Don't use any lambda functions in your processors or they won't be able to pickle.

pickle.dump(ll, open(path/'ld.pkl', 'wb'))

ll = pickle.load(open(path/'ld.pkl', 'rb'))

# ## Batching

# We have a bit of work to convert our `LabelList` in a `DataBunch` as we don't just want batches of IMDB reviews. We want to stream through all the texts concatenated. We also have to prepare the targets that are the newt words in the text. All of this is done with the next object called `LM_PreLoader`. At the beginning of each epoch, it'll shuffle the articles (if `shuffle=True`) and create a big stream by concatenating all of them. We divide this big stream in `bs` smaller streams. That we will read in chunks of bptt length.

# [Jump_to lesson 12 video](https://course.fast.ai/videos/?lesson=12&t=5565)

# Just using those for illustration purposes, they're not used otherwise.
from IPython.display import display,HTML
import pandas as pd

# Let's say our stream is:

stream = """
In this notebook, we will go back over the example of classifying movie reviews we studied in part 1 and dig deeper under the surface. 
First we will look at the processing steps necessary to convert text into numbers and how to customize it. By doing this, we'll have another example of the Processor used in the data block API.
Then we will study how we build a language model and train it.\n
"""
tokens = np.array(tp([stream])[0])

# Then if we split it in 6 batches it would give something like this:

bs,seq_len = 6,15
d_tokens = np.array([tokens[i*seq_len:(i+1)*seq_len] for i in range(bs)])
df = pd.DataFrame(d_tokens)
display(HTML(df.to_html(index=False,header=None)))

# Then if we have a `bptt` of 5, we would go over those three batches.

bs,bptt = 6,5
for k in range(3):
    d_tokens = np.array([tokens[i*seq_len + k*bptt:i*seq_len + (k+1)*bptt] for i in range(bs)])
    df = pd.DataFrame(d_tokens)
    display(HTML(df.to_html(index=False,header=None)))


#export
class LM_PreLoader():
    def __init__(self, data, bs=64, bptt=70, shuffle=False):
        self.data,self.bs,self.bptt,self.shuffle = data,bs,bptt,shuffle
        total_len = sum([len(t) for t in data.x])
        self.n_batch = total_len // bs
        self.batchify()
    
    def __len__(self): return ((self.n_batch-1) // self.bptt) * self.bs
    
    def __getitem__(self, idx):
        source = self.batched_data[idx % self.bs]
        seq_idx = (idx // self.bs) * self.bptt
        return source[seq_idx:seq_idx+self.bptt],source[seq_idx+1:seq_idx+self.bptt+1]
    
    def batchify(self):
        texts = self.data.x
        if self.shuffle: texts = texts[torch.randperm(len(texts))]
        stream = torch.cat([tensor(t) for t in texts])
        self.batched_data = stream[:self.n_batch * self.bs].view(self.bs, self.n_batch)


dl = DataLoader(LM_PreLoader(ll.valid, shuffle=True), batch_size=64)

# Let's check it all works ok: `x1`, `y1`, `x2` and `y2` should all be of size `bs`  by `bptt`. The texts in each row of `x1` should continue in `x2`. `y1` and `y2` should have the same texts as their `x` counterpart, shifted of one position to the right.

iter_dl = iter(dl)
x1,y1 = next(iter_dl)
x2,y2 = next(iter_dl)

x1.size(),y1.size()

vocab = proc_num.vocab

" ".join(vocab[o] for o in x1[0])

" ".join(vocab[o] for o in y1[0])

" ".join(vocab[o] for o in x2[0])


# And let's prepare some convenience function to do this quickly.

# +
#export
def get_lm_dls(train_ds, valid_ds, bs, bptt, **kwargs):
    return (DataLoader(LM_PreLoader(train_ds, bs, bptt, shuffle=True), batch_size=bs, **kwargs),
            DataLoader(LM_PreLoader(valid_ds, bs, bptt, shuffle=False), batch_size=2*bs, **kwargs))

def lm_databunchify(sd, bs, bptt, **kwargs):
    return DataBunch(*get_lm_dls(sd.train, sd.valid, bs, bptt, **kwargs))


# -

bs,bptt = 64,70
data = lm_databunchify(ll, bs, bptt)

# ## Batching for classification

# When we will want to tackle classification, gathering the data will be a bit different: first we will label our texts with the folder they come from, and then we will need to apply padding to batch them together. To avoid mixing very long texts with very short ones, we will also use `Sampler` to sort (with a bit of randomness for the training set) our samples by length.
#
# First the data block API calls shold look familiar.

# [Jump_to lesson 12 video](https://course.fast.ai/videos/?lesson=12&t=5877)

proc_cat = CategoryProcessor()

il = TextList.from_files(path, include=['train', 'test'])
sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='test'))
ll = label_by_func(sd, parent_labeler, proc_x = [proc_tok, proc_num], proc_y=proc_cat)

pickle.dump(ll, open(path/'ll_clas.pkl', 'wb'))

ll = pickle.load(open(path/'ll_clas.pkl', 'rb'))

# Let's check the labels seem consistent with the texts.

[(ll.train.x_obj(i), ll.train.y_obj(i)) for i in [1,12552]]

# We saw samplers in notebook 03. For the validation set, we will simply sort the samples by length, and we begin with the longest ones for memory reasons (it's better to always have the biggest tensors first).

# +
#export
from torch.utils.data import Sampler

class SortSampler(Sampler):
    def __init__(self, data_source, key): self.data_source,self.key = data_source,key
    def __len__(self): return len(self.data_source)
    def __iter__(self):
        return iter(sorted(list(range(len(self.data_source))), key=self.key, reverse=True))


# -

# For the training set, we want some kind of randomness on top of this. So first, we shuffle the texts and build megabatches of size `50 * bs`. We sort those megabatches by length before splitting them in 50 minibatches. That way we will have randomized batches of roughly the same length.
#
# Then we make sure to have the biggest batch first and shuffle the order of the other batches. We also make sure the last batch stays at the end because its size is probably lower than batch size.

#export
class SortishSampler(Sampler):
    def __init__(self, data_source, key, bs):
        self.data_source,self.key,self.bs = data_source,key,bs

    def __len__(self) -> int: return len(self.data_source)

    def __iter__(self):
        idxs = torch.randperm(len(self.data_source))
        megabatches = [idxs[i:i+self.bs*50] for i in range(0, len(idxs), self.bs*50)]
        sorted_idx = torch.cat([tensor(sorted(s, key=self.key, reverse=True)) for s in megabatches])
        batches = [sorted_idx[i:i+self.bs] for i in range(0, len(sorted_idx), self.bs)]
        max_idx = torch.argmax(tensor([self.key(ck[0]) for ck in batches]))  # find the chunk with the largest key,
        batches[0],batches[max_idx] = batches[max_idx],batches[0]            # then make sure it goes first.
        batch_idxs = torch.randperm(len(batches)-2)
        sorted_idx = torch.cat([batches[i+1] for i in batch_idxs]) if len(batches) > 1 else LongTensor([])
        sorted_idx = torch.cat([batches[0], sorted_idx, batches[-1]])
        return iter(sorted_idx)


# Padding: we had the padding token (that as an id of 1) at the end of each sequence to make them all the same size when batching them. Note that we need padding at the end to be able to use `PyTorch` convenience functions that will let us ignore that padding (see 12c).

#export
def pad_collate(samples, pad_idx=1, pad_first=False):
    max_len = max([len(s[0]) for s in samples])
    res = torch.zeros(len(samples), max_len).long() + pad_idx
    for i,s in enumerate(samples):
        if pad_first: res[i, -len(s[0]):] = LongTensor(s[0])
        else:         res[i, :len(s[0]) ] = LongTensor(s[0])
    return res, tensor([s[1] for s in samples])


bs = 64
train_sampler = SortishSampler(ll.train.x, key=lambda t: len(ll.train[int(t)][0]), bs=bs)
train_dl = DataLoader(ll.train, batch_size=bs, sampler=train_sampler, collate_fn=pad_collate)

iter_dl = iter(train_dl)
x,y = next(iter_dl)

lengths = []
for i in range(x.size(0)): lengths.append(x.size(1) - (x[i]==1).sum().item())
lengths[:5], lengths[-1]

# The last one is the minimal length. This is the first batch so it has the longest sequence, but if look at the next one that is more random, we see lengths are roughly the sames.

x,y = next(iter_dl)
lengths = []
for i in range(x.size(0)): lengths.append(x.size(1) - (x[i]==1).sum().item())
lengths[:5], lengths[-1]

# We can see the padding at the end:

x


# And we add a convenience function:

# +
#export
def get_clas_dls(train_ds, valid_ds, bs, **kwargs):
    train_sampler = SortishSampler(train_ds.x, key=lambda t: len(train_ds.x[t]), bs=bs)
    valid_sampler = SortSampler(valid_ds.x, key=lambda t: len(valid_ds.x[t]))
    return (DataLoader(train_ds, batch_size=bs, sampler=train_sampler, collate_fn=pad_collate, **kwargs),
            DataLoader(valid_ds, batch_size=bs*2, sampler=valid_sampler, collate_fn=pad_collate, **kwargs))

def clas_databunchify(sd, bs, **kwargs):
    return DataBunch(*get_clas_dls(sd.train, sd.valid, bs, **kwargs))


# -

bs,bptt = 64,70
data = clas_databunchify(ll, bs)

# ## Export

# !python notebook2script.py 12_text.ipynb


