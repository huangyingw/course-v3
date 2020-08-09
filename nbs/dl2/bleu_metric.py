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

from fastai.text import *

# ## What is the BLEU metric?

# The BLEU metric has been introduced in [this article](https://www.aclweb.org/anthology/P02-1040) to come with some kind of way to evaluate the performance of translation models. It's based on the precision you hit with n-grams in your prediction compared to your target. Let's see this on an example. Imagine you have the target sentence
# ```
# the cat is walking in the garden
# ```
# and your model gives you the following output
# ```
# the cat is running in the fields
# ```
# We are going to compute the precision, which is the number of correctly predicted n-grams divided by the number of predicted n-grams for n going from 1 to 4.
#
# For 1-grams (or tokens, more simply), we have predicted 5 correct words out of 7, so we get a precision of 5/7. Note that the order doesn't matter, for instance predicting
# ```
# she read the book because she was interested in world history
# ```
# instead of
# ```
# she was interested in world history because she read the book
# ```
# would give a precision of 1 for the 1-grams.
#
# For 2-grams, in the first example, we have 3 correct 2-grams ("the cat", "cat is" and "in the") out of 6, so a precision of 3/6. In the second example, the precision for 2-grams is 9/10.
#
# For 3-grams, in the first example, we have only 1 correct 3-gram ("the cat is") out of 5, so a precision of 1/5. In the second example, the precision for 3-grams is 6/9.
#
# For 4-grams, in the first example, we don't have any 4-gram that is correct, so the precision is 0. In the second example, it's 4/8.
#
# There is one big drawback in just taking the raw precision: very often a seq2seq model will predict the same easy word. If take the prediction
# ```
# the the the the the the the the
# ```
# for our first example, it would score a precision of 1. for 1-grams (because `the` is in the target sentence, so all the words are considered correct). To avoid that, we put a maximum for a given words to the number of times it can be considered correct, which is the number of times that word is in the target sentence. So in our example, only 2 of the 7 the are considered correct and this clamped precision gives us 2/7 for 1-grams.
#
# One thing to note is that when we deal with a whole corpus, we take the sum of all the corrects over all the sentences then divide by the sum of all the predicted over all the sentences (we don't average precisions over each sentence).
#
# To compute the BLEU score, the final formula is then
# ```
# BLEU = length_penalty * ((p1 * p2 * p3 * p4) ** 0.25)
# ```
# which is the geometric average of `p1`, `p2`, `p3` and `p4` (our n-gram precision scores) multiplied by a penalty given for the length: we penalize longer predictions with the precision scores, but short ones get it easier, especially if they only contain correct words. So we apply the following penalty:
# ```
# length_penalty = 1 if len(pred) >= len(targ) else (1 - exp(len(targ)/len(pred)))
# ```
#
# And if we are taking the BLEU score of a whole corpus, we use the sum of the lengths of predicted sentences and the sum of the lengths of predicted targets.

# ## Let's code this

# There is an implementation of BLEU in nltk, but the problem is that it's designed to support lists of tokenized texts, and therefore is very slow (5 hours announced on the validation set of the translation notebook for the average of BLEU scores of sentences). We have numericalized text, so it's easier to reimplement this and only deal with integers.
#
# Specifically we are going to use the Counter class, which is going to count the number of instances of each n-gram in the predicted sentence and the target one:

targ = [1,2,3,4,5,1,2]
pred = [1,2,3,7,5,1,1]

cnt_pred,cnt_targ = Counter(pred),Counter(targ)

# Now the number of corrects is the number of words in `pred` that are in `targ`, with a cap that is the number of times they appear in `targ`.

corrects = sum([min(c, cnt_targ[g]) for g,c in cnt_pred.items()])
corrects


# Now this works very well for ints, but it won't work for a list of ints, since a `Counter` requires the objects inside to be hashable. That's why we define a custom class:

class NGram():
    def __init__(self, ngram, max_n=5000): self.ngram,self.max_n = ngram,max_n
    def __eq__(self, other):
        if len(self.ngram) != len(other.ngram): return False
        return np.all(np.array(self.ngram) == np.array(other.ngram))
    def __hash__(self): return int(sum([o * self.max_n**i for i,o in enumerate(self.ngram)]))


# `max_n` should be set to the vocab size, so we're sure we don't have two different ngrams with the same hash (it needs to be an int).
#
# Then we can define `get_grams` to gather all the possible ngrams:

def get_grams(x, n, max_n=5000):
    return x if n==1 else [NGram(x[i:i+n], max_n=max_n) for i in range(len(x)-n+1)]


# From here, it's easy to compute correctly predicted ngrams:

def get_correct_ngrams(pred, targ, n, max_n=5000):
    pred_grams,targ_grams = get_grams(pred, n, max_n=max_n),get_grams(targ, n, max_n=max_n)
    pred_cnt,targ_cnt = Counter(pred_grams),Counter(targ_grams)
    return sum([min(c, targ_cnt[g]) for g,c in pred_cnt.items()]),len(pred_grams)


# The BLEU metric over two sentences can be written as:

def sentence_bleu(pred, targ, max_n=5000):
    corrects = [get_correct_ngrams(pred,targ,n,max_n=max_n) for n in range(1,5)]
    n_precs = [c/t for c,t in corrects]
    len_penalty = exp(1 - len(targ)/len(pred)) if len(pred) < len(targ) else 1
    return len_penalty * ((n_precs[0]*n_precs[1]*n_precs[2]*n_precs[3]) ** 0.25)


# And the BLEU metric over a corpus is:

def corpus_bleu(preds, targs, max_n=5000):
    pred_len,targ_len,n_precs,counts = 0,0,[0]*4,[0]*4
    for pred,targ in zip(preds,targs):
        pred_len += len(pred)
        targ_len += len(targ)
        for i in range(4):
            c,t = ngram_corrects(pred, targ, i+1, max_n=max_n)
            n_precs[i] += c
            counts[i] += t
    n_precs = [c/t for c,t in zip(n_precs,counts)]
    len_penalty = exp(1 - targ_len/pred_len) if pred_len < targ_len else 1
    return len_penalty * ((n_precs[0]*n_precs[1]*n_precs[2]*n_precs[3]) ** 0.25)


# This takes 11s to run on our validation set (instead of 5 hours), so we can even use it as a metric during training, we have to define it as a `Callback`

class CorpusBLEU(Callback):
    def __init__(self, vocab_sz):
        self.vocab_sz = vocab_sz
        self.name = 'bleu'
    
    def on_epoch_begin(self, **kwargs):
        self.pred_len,self.targ_len,self.n_precs,self.counts = 0,0,[0]*4,[0]*4
    
    def on_batch_end(self, last_output, last_target, **kwargs):
        last_output = last_output.argmax(dim=-1)
        for pred,targ in zip(last_output.cpu().numpy(),last_target.cpu().numpy()):
            self.pred_len += len(pred)
            self.targ_len += len(targ)
            for i in range(4):
                c,t = get_correct_ngrams(pred, targ, i+1, max_n=self.vocab_sz)
                self.n_precs[i] += c
                self.counts[i] += t
    
    def on_epoch_end(self, last_metrics, **kwargs):
        n_precs = [c/t for c,t in zip(n_precs,counts)]
        len_penalty = exp(1 - targ_len/pred_len) if pred_len < targ_len else 1
        bleu = len_penalty * ((n_precs[0]*n_precs[1]*n_precs[2]*n_precs[3]) ** 0.25)
        return add_metrics(last_metrics, bleu)
