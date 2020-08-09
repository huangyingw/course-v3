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

# # ULMFit

# +
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline
# -

#export
from exp.nb_12a import *

# ## Data

# We load the data from 12a, instructions to create that file are there if you don't have it yet so go ahead and see.

# [Jump_to lesson 12 video](https://course.fast.ai/videos/?lesson=12&t=7459)

path = datasets.untar_data(datasets.URLs.IMDB)

ll = pickle.load(open(path/'ll_lm.pkl', 'rb'))

bs,bptt = 128,70
data = lm_databunchify(ll, bs, bptt)

vocab = ll.train.proc_x[1].vocab

# ## Finetuning the LM

# Before tackling the classification task, we have to finetune our language model to the IMDB corpus.

# We have pretrained a small model on [wikitext 103](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/) that you can download by uncommenting the following cell.

# +
# # ! wget http://files.fast.ai/models/wt103_tiny.tgz -P {path}
# # ! tar xf {path}/wt103_tiny.tgz -C {path}
# -

dps = tensor([0.1, 0.15, 0.25, 0.02, 0.2]) * 0.5
tok_pad = vocab.index(PAD)

emb_sz, nh, nl = 300, 300, 2
model = get_language_model(len(vocab), emb_sz, nh, nl, tok_pad, *dps)

old_wgts  = torch.load(path/'pretrained'/'pretrained.pth')
old_vocab = pickle.load(open(path/'pretrained'/'vocab.pkl', 'rb'))

# In our current vocabulary, it is very unlikely that the ids correspond to what is in the vocabulary used to train the pretrain model. The tokens are sorted by frequency (apart from the special tokens that are all first) so that order is specific to the corpus used. For instance, the word 'house' has different ids in the our current vocab and the pretrained one.

idx_house_new, idx_house_old = vocab.index('house'),old_vocab.index('house')

# We somehow need to match our pretrained weights to the new vocabulary. This is done on the embeddings and the decoder (since the weights between embeddings and decoders are tied) by putting the rows of the embedding matrix (or decoder bias) in the right order.
#
# It may also happen that we have words that aren't in the pretrained vocab, in this case, we put the mean of the pretrained embedding weights/decoder bias.

house_wgt  = old_wgts['0.emb.weight'][idx_house_old]
house_bias = old_wgts['1.decoder.bias'][idx_house_old] 


def match_embeds(old_wgts, old_vocab, new_vocab):
    wgts = old_wgts['0.emb.weight']
    bias = old_wgts['1.decoder.bias']
    wgts_m,bias_m = wgts.mean(dim=0),bias.mean()
    new_wgts = wgts.new_zeros(len(new_vocab), wgts.size(1))
    new_bias = bias.new_zeros(len(new_vocab))
    otoi = {v:k for k,v in enumerate(old_vocab)}
    for i,w in enumerate(new_vocab): 
        if w in otoi:
            idx = otoi[w]
            new_wgts[i],new_bias[i] = wgts[idx],bias[idx]
        else: new_wgts[i],new_bias[i] = wgts_m,bias_m
    old_wgts['0.emb.weight']    = new_wgts
    old_wgts['0.emb_dp.emb.weight'] = new_wgts
    old_wgts['1.decoder.weight']    = new_wgts
    old_wgts['1.decoder.bias']      = new_bias
    return old_wgts


wgts = match_embeds(old_wgts, old_vocab, vocab)

# Now let's check that the word "*house*" was properly converted.

test_near(wgts['0.emb.weight'][idx_house_new],house_wgt)
test_near(wgts['1.decoder.bias'][idx_house_new],house_bias)

# We can load the pretrained weights in our model before beginning training.

model.load_state_dict(wgts)

# If we want to apply discriminative learning rates, we need to split our model in different layer groups. Let's have a look at our model.

model


# Then we split by doing two groups for each rnn/corresponding dropout, then one last group that contains the embeddings/decoder. This is the one that needs to be trained the most as we may have new embeddings vectors.

def lm_splitter(m):
    groups = []
    for i in range(len(m[0].rnns)): groups.append(nn.Sequential(m[0].rnns[i], m[0].hidden_dps[i]))
    groups += [nn.Sequential(m[0].emb, m[0].emb_dp, m[0].input_dp, m[1])]
    return [list(o.parameters()) for o in groups]


# First we train with the RNNs freezed.

for rnn in model[0].rnns:
    for p in rnn.parameters(): p.requires_grad_(False)

cbs = [partial(AvgStatsCallback,accuracy_flat),
       CudaCallback, Recorder,
       partial(GradientClipping, clip=0.1),
       partial(RNNTrainer, α=2., β=1.),
       ProgressCallback]

learn = Learner(model, data, cross_entropy_flat, opt_func=adam_opt(),
                cb_funcs=cbs, splitter=lm_splitter)

lr = 2e-2
cbsched = sched_1cycle([lr], pct_start=0.5, mom_start=0.8, mom_mid=0.7, mom_end=0.8)

learn.fit(1, cbs=cbsched)

# Then the whole model with discriminative learning rates.

for rnn in model[0].rnns:
    for p in rnn.parameters(): p.requires_grad_(True)

lr = 2e-3
cbsched = sched_1cycle([lr/2., lr/2., lr], pct_start=0.5, mom_start=0.8, mom_mid=0.7, mom_end=0.8)

learn.fit(10, cbs=cbsched)

# We only need to save the encoder (first part of the model) for the classification, as well as the vocabulary used (we will need to use the same in the classification task).

torch.save(learn.model[0].state_dict(), path/'finetuned_enc.pth')

pickle.dump(vocab, open(path/'vocab_lm.pkl', 'wb'))

torch.save(learn.model.state_dict(), path/'finetuned.pth')

# ## Classifier

# We have to process the data again otherwise pickle will complain. We also have to use the same vocab as the language model.

# [Jump_to lesson 12 video](https://course.fast.ai/videos/?lesson=12&t=7554)

vocab = pickle.load(open(path/'vocab_lm.pkl', 'rb'))
proc_tok,proc_num,proc_cat = TokenizeProcessor(),NumericalizeProcessor(vocab=vocab),CategoryProcessor()

il = TextList.from_files(path, include=['train', 'test'])
sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='test'))
ll = label_by_func(sd, parent_labeler, proc_x = [proc_tok, proc_num], proc_y=proc_cat)

pickle.dump(ll, open(path/'ll_clas.pkl', 'wb'))

ll = pickle.load(open(path/'ll_clas.pkl', 'rb'))
vocab = pickle.load(open(path/'vocab_lm.pkl', 'rb'))

bs,bptt = 64,70
data = clas_databunchify(ll, bs)

# ### Ignore padding

# We will those two utility functions from PyTorch to ignore the padding in the inputs.

#export
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Let's see how this works: first we grab a batch of the training set.

x,y = next(iter(data.train_dl))

x.size()

# We need to pass to the utility functions the lengths of our sentences because it's applied after the embedding, so we can't see the padding anymore.

lengths = x.size(1) - (x == 1).sum(1)
lengths[:5]

tst_emb = nn.Embedding(len(vocab), 300)

tst_emb(x).shape

128*70

# We create a `PackedSequence` object that contains all of our unpadded sequences

packed = pack_padded_sequence(tst_emb(x), lengths, batch_first=True)

packed

packed.data.shape

len(packed.batch_sizes)

8960//70

# This object can be passed to any RNN directly while retaining the speed of CuDNN.

tst = nn.LSTM(300, 300, 2)

y,h = tst(packed)

# Then we can unpad it with the following function for other modules:

unpack = pad_packed_sequence(y, batch_first=True)

unpack[0].shape

unpack[1]


# We need to change our model a little bit to use this.

#export
class AWD_LSTM1(nn.Module):
    "AWD-LSTM inspired by https://arxiv.org/abs/1708.02182."
    initrange=0.1

    def __init__(self, vocab_sz, emb_sz, n_hid, n_layers, pad_token,
                 hidden_p=0.2, input_p=0.6, embed_p=0.1, weight_p=0.5):
        super().__init__()
        self.bs,self.emb_sz,self.n_hid,self.n_layers,self.pad_token = 1,emb_sz,n_hid,n_layers,pad_token
        self.emb = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.emb_dp = EmbeddingDropout(self.emb, embed_p)
        self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz), 1,
                             batch_first=True) for l in range(n_layers)]
        self.rnns = nn.ModuleList([WeightDropout(rnn, weight_p) for rnn in self.rnns])
        self.emb.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = RNNDropout(input_p)
        self.hidden_dps = nn.ModuleList([RNNDropout(hidden_p) for l in range(n_layers)])

    def forward(self, input):
        bs,sl = input.size()
        mask = (input == self.pad_token)
        lengths = sl - mask.long().sum(1)
        n_empty = (lengths == 0).sum()
        if n_empty > 0:
            input = input[:-n_empty]
            lengths = lengths[:-n_empty]
            self.hidden = [(h[0][:,:input.size(0)], h[1][:,:input.size(0)]) for h in self.hidden]
        raw_output = self.input_dp(self.emb_dp(input))
        new_hidden,raw_outputs,outputs = [],[],[]
        for l, (rnn,hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output = pack_padded_sequence(raw_output, lengths, batch_first=True)
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            raw_output = pad_packed_sequence(raw_output, batch_first=True)[0]
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1: raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
            new_hidden.append(new_h)
        self.hidden = to_detach(new_hidden)
        return raw_outputs, outputs, mask

    def _one_hidden(self, l):
        "Return one hidden state."
        nh = self.n_hid if l != self.n_layers - 1 else self.emb_sz
        return next(self.parameters()).new(1, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        self.hidden = [(self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)]


# ### Concat pooling

# We will use three things for the classification head of the model: the last hidden state, the average of all the hidden states and the maximum of all the hidden states. The trick is just to, once again, ignore the padding in the last element/average/maximum.

# [Jump_to lesson 12 video](https://course.fast.ai/videos/?lesson=12&t=7604)

class Pooling(nn.Module):
    def forward(self, input):
        raw_outputs,outputs,mask = input
        output = outputs[-1]
        lengths = output.size(1) - mask.long().sum(dim=1)
        avg_pool = output.masked_fill(mask[:,:,None], 0).sum(dim=1)
        avg_pool.div_(lengths.type(avg_pool.dtype)[:,None])
        max_pool = output.masked_fill(mask[:,:,None], -float('inf')).max(dim=1)[0]
        x = torch.cat([output[torch.arange(0, output.size(0)),lengths-1], max_pool, avg_pool], 1) #Concat pooling.
        return output,x


emb_sz, nh, nl = 300, 300, 2
tok_pad = vocab.index(PAD)

enc = AWD_LSTM1(len(vocab), emb_sz, n_hid=nh, n_layers=nl, pad_token=tok_pad)
pool = Pooling()
enc.bs = bs
enc.reset()

x,y = next(iter(data.train_dl))
output,c = pool(enc(x))

# We can check we have padding with 1s at the end of each text (except the first which is the longest).

x

# PyTorch puts 0s everywhere we had padding in the `output` when unpacking.

test_near((output.sum(dim=2) == 0).float(), (x==tok_pad).float())

# So the last hidden state isn't the last element of `output`. Let's check we got everything right. 

for i in range(bs):
    length = x.size(1) - (x[i]==1).long().sum()
    out_unpad = output[i,:length]
    test_near(out_unpad[-1], c[i,:300])
    test_near(out_unpad.max(0)[0], c[i,300:600])
    test_near(out_unpad.mean(0), c[i,600:])


# Our pooling layer properly ignored the padding, so now let's group it with a classifier.

def bn_drop_lin(n_in, n_out, bn=True, p=0., actn=None):
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers


class PoolingLinearClassifier(nn.Module):
    "Create a linear classifier with pooling."

    def __init__(self, layers, drops):
        super().__init__()
        mod_layers = []
        activs = [nn.ReLU(inplace=True)] * (len(layers) - 2) + [None]
        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activs):
            mod_layers += bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*mod_layers)

    def forward(self, input):
        raw_outputs,outputs,mask = input
        output = outputs[-1]
        lengths = output.size(1) - mask.long().sum(dim=1)
        avg_pool = output.masked_fill(mask[:,:,None], 0).sum(dim=1)
        avg_pool.div_(lengths.type(avg_pool.dtype)[:,None])
        max_pool = output.masked_fill(mask[:,:,None], -float('inf')).max(dim=1)[0]
        x = torch.cat([output[torch.arange(0, output.size(0)),lengths-1], max_pool, avg_pool], 1) #Concat pooling.
        x = self.layers(x)
        return x


# Then we just have to feed our texts to those two blocks, (but we can't give them all at once to the AWD_LSTM or we might get OOM error: we'll go for chunks of bptt length to regularly detach the history of our hidden states.)

def pad_tensor(t, bs, val=0.):
    if t.size(0) < bs:
        return torch.cat([t, val + t.new_zeros(bs-t.size(0), *t.shape[1:])])
    return t


class SentenceEncoder(nn.Module):
    def __init__(self, module, bptt, pad_idx=1):
        super().__init__()
        self.bptt,self.module,self.pad_idx = bptt,module,pad_idx

    def concat(self, arrs, bs):
        return [torch.cat([pad_tensor(l[si],bs) for l in arrs], dim=1) for si in range(len(arrs[0]))]
    
    def forward(self, input):
        bs,sl = input.size()
        self.module.bs = bs
        self.module.reset()
        raw_outputs,outputs,masks = [],[],[]
        for i in range(0, sl, self.bptt):
            r,o,m = self.module(input[:,i: min(i+self.bptt, sl)])
            masks.append(pad_tensor(m, bs, 1))
            raw_outputs.append(r)
            outputs.append(o)
        return self.concat(raw_outputs, bs),self.concat(outputs, bs),torch.cat(masks,dim=1)


def get_text_classifier(vocab_sz, emb_sz, n_hid, n_layers, n_out, pad_token, bptt, output_p=0.4, hidden_p=0.2, 
                        input_p=0.6, embed_p=0.1, weight_p=0.5, layers=None, drops=None):
    "To create a full AWD-LSTM"
    rnn_enc = AWD_LSTM1(vocab_sz, emb_sz, n_hid=n_hid, n_layers=n_layers, pad_token=pad_token,
                        hidden_p=hidden_p, input_p=input_p, embed_p=embed_p, weight_p=weight_p)
    enc = SentenceEncoder(rnn_enc, bptt)
    if layers is None: layers = [50]
    if drops is None:  drops = [0.1] * len(layers)
    layers = [3 * emb_sz] + layers + [n_out] 
    drops = [output_p] + drops
    return SequentialRNN(enc, PoolingLinearClassifier(layers, drops))


emb_sz, nh, nl = 300, 300, 2
dps = tensor([0.4, 0.3, 0.4, 0.05, 0.5]) * 0.25
model = get_text_classifier(len(vocab), emb_sz, nh, nl, 2, 1, bptt, *dps)


# ### Training

# We load our pretrained encoder and freeze it.

# [Jump_to lesson 12 video](https://course.fast.ai/videos/?lesson=12&t=7684)

def class_splitter(m):
    enc = m[0].module
    groups = [nn.Sequential(enc.emb, enc.emb_dp, enc.input_dp)]
    for i in range(len(enc.rnns)): groups.append(nn.Sequential(enc.rnns[i], enc.hidden_dps[i]))
    groups.append(m[1])
    return [list(o.parameters()) for o in groups]


for p in model[0].parameters(): p.requires_grad_(False)

cbs = [partial(AvgStatsCallback,accuracy),
       CudaCallback, Recorder,
       partial(GradientClipping, clip=0.1),
       ProgressCallback]

model[0].module.load_state_dict(torch.load(path/'finetuned_enc.pth'))

learn = Learner(model, data, F.cross_entropy, opt_func=adam_opt(), cb_funcs=cbs, splitter=class_splitter)

lr = 1e-2
cbsched = sched_1cycle([lr], mom_start=0.8, mom_mid=0.7, mom_end=0.8)

learn.fit(1, cbs=cbsched)

for p in model[0].module.rnns[-1].parameters(): p.requires_grad_(True)

lr = 5e-3
cbsched = sched_1cycle([lr/2., lr/2., lr/2., lr], mom_start=0.8, mom_mid=0.7, mom_end=0.8)

learn.fit(1, cbs=cbsched)

for p in model[0].parameters(): p.requires_grad_(True)

lr = 1e-3
cbsched = sched_1cycle([lr/8., lr/4., lr/2., lr], mom_start=0.8, mom_mid=0.7, mom_end=0.8)

learn.fit(2, cbs=cbsched)

x,y = next(iter(data.valid_dl))

# Predicting on the padded batch or on the individual unpadded samples give the same results.

pred_batch = learn.model.eval()(x.cuda())

pred_ind = []
for inp in x:
    length = x.size(1) - (inp == 1).long().sum()
    inp = inp[:length]
    pred_ind.append(learn.model.eval()(inp[None].cuda()))

assert near(pred_batch, torch.cat(pred_ind))


