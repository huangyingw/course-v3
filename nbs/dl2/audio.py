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

# # Audio classification the from-scratch way

# Thanks to the SF Study Group practitioners & participants in the [Deep Learning for Audio fastai forum thread](https://forums.fast.ai/t/deep-learning-with-audio-thread/), especially: [Molly Beavers](https://forums.fast.ai/u/marii/summary), [Simon Grest](https://forums.fast.ai/u/simonjhb/), [Stefano Giomo](https://forums.fast.ai/u/ste), [Thom Mackey](https://forums.fast.ai/u/ThomM), [Zach Caceres](https://forums.fast.ai/u/zachcaceres), [Harry Coultas Blum](https://forums.fast.ai/u/baz), & [Robert Bracco](https://forums.fast.ai/u/madeupmasters).

# We're going to demonstrate the technique of classifying audio samples by first converting the audio into spectrograms, then treating the spectrograms as images. Once we've converted the spectrograms to images, the workflow is just the same as using imagenette or any other image classification task.
#
# What do we need to do?
# * Download the data
# * Load the data 
# * Transform the data into spectrograms
# * Load the audio data into a databunch such that we can use our previously-defined `learner` object
#
# Still to come - 1D convolutional models, RNNs with audio… and more, with your contribution :)

# ### Setup & imports

# %matplotlib inline

# We rely heavily on [torchaudio](https://github.com/pytorch/audio) - which you'll have to compile to install.

#export
from exp.nb_12a import *
import torchaudio
from torchaudio import transforms

#export
AUDIO_EXTS = {str.lower(k) for k,v in mimetypes.types_map.items() if v.startswith('audio/')}

# ### Download

# This should be one line; it's only so complicated because the target .tgz file doesn't extract itself to its own directory.

dsid = "ST-AEDS-20180100_1-OS"
data_url = f'http://www.openslr.org/resources/45/{dsid}' # actual URL has .tgz extension but untar_data doesn't like that
path = Path.home() / Path(f".fastai/data/{dsid}/")
datasets.untar_data(data_url, dest=path)
path

# ## Loading into an AudioList

# Getting a file list the `08_data_block` way.

# The "manual" way using `get_files`…

audios = get_files(path, extensions=AUDIO_EXTS)
print(f"Found {len(audios)} audio files")
audios[:5]


# …But that's not very exciting. Let's make an `AudioList`, so we can use transforms, and define how to `get` an Audio.

# ### AudioList

#export
class AudioList(ItemList):
    @classmethod
    def from_files(cls, path, extensions=None, recurse=True, include=None, **kwargs):
        if extensions is None: extensions = AUDIO_EXTS
        return cls(get_files(path, extensions, recurse=recurse, include=include), path, **kwargs)
    
    def get(self, fn): 
        sig, sr = torchaudio.load(fn)
        assert sig.size(0) == 1, "Non-mono audio detected, mono only supported for now."
        return (sig, sr)


al = AudioList.from_files(path); al

# It looks like this is full of file paths, but that's just the `repr` talking. Actually accessing an item from the list calls the `get` method and returns a `(Tensor, Int)` tuple representing the signal & sample rate.

al[0]

# ## Splitting into train/validation

# Our data is all in one folder, there's no specific validation set, so let's just split it at random.

sd = SplitData.split_by_func(al, partial(random_splitter, p_valid=0.2))

sd


# ## Labeling

# Our labels are encoded in our filenames. For example, `m0003_us_m0003_00032.wav` has the label `m0003`. Let's make a regex labeler, then use it.

#export
def re_labeler(fn, pat): return re.findall(pat, str(fn))[0]


label_pat = r'/([mf]\d+)_'
speaker_labeler = partial(re_labeler, pat=label_pat)
ll = label_by_func(sd, speaker_labeler, proc_y=CategoryProcessor())

ll

# ## Transforms: audio clipping & conversion to spectrograms

# The pytorch dataloader needs to be all tensors to be the same size, but our input audio files are of different sizes, so we need to trim them. Also, recall that we're not going to send the model the audio signal directly; we're going to convert it to spectrograms first. We can treat these steps as transforms. We can also apply data augmentation to both the signal and the spectrograms. 

# Small helpers to show some audio & compare transformed versions.

#export
from IPython.display import Audio
def show_audio(ad):
    sig,sr=ad
    display(Audio(data=sig, rate=sr))


#export
def show_audio_in_out(orig, trans):
    """Helper to plot input and output signal in different colors"""
    osig,osr = orig
    tsig,tsr = trans
    print("↓ Original ↓")
    show_audio(orig)
    print("↓ Transformed ↓")
    show_audio(trans)
    if orig is not None: plt.plot(osig[0], 'm', label="Orig.")
    if trans is not None: plt.plot(tsig[0], 'c', alpha=0.5, label="Transf.")
    plt.legend()
    plt.show()


# ### toCuda

# The other transforms both use all-tensor ops, so it should help. Let's try it out.

#export
class ToCuda(Transform):
    _order=10
    def __call__(self, ad):
        sig,sr = ad
        return (sig.cuda(), sr)


ToCuda()(ll.train[0][0])


# ### PadOrTrim

# `torchaudio` has one for this already. All we're doing is taking an argument in milliseconds rather than frames.

#export
class PadOrTrim(Transform):
    _order=11
    def __init__(self,msecs): self.msecs = msecs
    def __call__(self, ad): 
        sig,sr = ad
        mx = sr//1000 * self.msecs
        return (transforms.PadTrim(mx)(sig), sr)


# *Note - this won't work if you've already run the notebook all the way through, because `ll` now contains Tensors representing Spectrograms, not `(Signal, SampleRate)` tuples.*

pt = PadOrTrim(3000) ## duration in milliseconds
show_audio_in_out(ll.train[0][0], pt(ll.train[0][0]))


# ### Signal shifting

# Signal shifting is a useful augmentation which can be applied to the signal itself; it simply shifts the input to the left or right by some %. Intuitively, this is useful for classification, but perhaps not for predictive models, as it can change the temporal order of things when the end of the signal wraps around to the front.
#
# This implementation lightly adapted from the [original implementation](https://github.com/mogwai/fastai-audio/blob/sg-cache/Google%20Small%20ASR.ipynb) by [Harry Coultas Blum](https://github.com/mogwai/).

#export
class SignalShift(Transform):
    _order=20
    def __init__(self, max_shift_pct=.6): self.max_shift_pct = max_shift_pct
    def __call__(self, ad):
        sig,sr = ad
        roll_by = int(random.random()*self.max_shift_pct*len(sig[0]))
        return (sig.roll(roll_by), sr)


shifter = SignalShift()
show_audio_in_out(ll.train[0][0], shifter(ll.train[0][0]))


# ### Spectrogram

# Once we're done augmenting the signal directly, we want to convert our 1D signal to a 2D [spectrogram](https://en.wikipedia.org/wiki/Spectrogram).
#
# Luckily, `torchaudio` has functions for calculation & conversion to spectrograms. Let's wrap their functions.

#export
class Spectrogrammer(Transform):
    _order=90
    def __init__(self, to_mel=True, to_db=True, n_fft=400, ws=None, hop=None, 
                 f_min=0.0, f_max=None, pad=0, n_mels=128, top_db=None, normalize=False):
        self.to_mel, self.to_db, self.n_fft, self.ws, self.hop, self.f_min, self.f_max, \
        self.pad, self.n_mels, self.top_db, self.normalize = to_mel, to_db, n_fft, \
        ws, hop, f_min, f_max, pad, n_mels, top_db, normalize

    def __call__(self, ad):
        sig,sr = ad
        if self.to_mel:
            spec = transforms.MelSpectrogram(sr, self.n_fft, self.ws, self.hop, self.f_min, 
                                             self.f_max, self.pad, self.n_mels)(sig)
        else: 
            spec = transforms.Spectrogram(self.n_fft, self.ws, self.hop, self.pad, 
                                          normalize=self.normalize)(sig)
        if self.to_db:
            spec = transforms.SpectrogramToDB(top_db=self.top_db)(spec)
        spec = spec.permute(0,2,1) # reshape so it looks good to humans
        return spec


# Small helper to show a spectrogram.

#export
def show_spectro(img, ax=None, figsize=(6,6), with_shape=True):
    if hasattr(img,"device") & str(img.device).startswith("cuda"): img = img.cpu()
    if ax is None: _,ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(img if (img.shape[0]==3) else img.squeeze(0))
    if with_shape: display(f'Tensor shape={img.shape}')


# *Note - this won't work if you've already run the notebook all the way through, because `ll` now contains Tensors representing Spectrograms, not `(Signal, SampleRate)` tuples.*

speccer = Spectrogrammer(to_db=True, n_fft=1024, n_mels=64, top_db=80)
show_spectro(speccer(ll.train[0][0]))


# ### SpecAugment

# An implementation of the [SpecAugment data augmentation technique](https://arxiv.org/abs/1904.08779) from Google Brain, which was shown to improve on SOTA results for audio classification. Effectively, it masks some number of contiguous frequency bands or timesteps. The intuition is similar to [cutout](https://arxiv.org/abs/1708.04552) for "normal" images. See their [blog post](https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html) for a good discussion.
#
# This implementation lightly adapted from both [Zach Caceres](https://github.com/zcaceres/) and [Harry Coultas Blum](https://github.com/mogwai/) & [Robert Bracco](https://forums.fast.ai/u/madeupmasters). Rather than static `F` or `T` parameters for a maximum width of the masks per the paper, we're taking a "maximum mask %" and calculating the `F` and `T` from that.
#
# This does not attempt to implement the "time warping" augmentation from the paper, as it is not directly supported by pytorch, and provides only modest gains for considerable computational cost. [Zach Caceres's SpecAugment implementation](https://github.com/zcaceres/spec_augment) provides a full implementation of time warping.

#export
class SpecAugment(Transform):
    _order=99
    def __init__(self, max_mask_pct=0.2, freq_masks=1, time_masks=1, replace_with_zero=False):
        self.max_mask_pct, self.freq_masks, self.time_masks, self.replace_with_zero = \
        max_mask_pct, freq_masks, time_masks, replace_with_zero
        if not 0 <= self.max_mask_pct <= 1.0: 
            raise ValueError( f"max_mask_pct must be between 0.0 and 1.0, but it's {self.max_mask_pct}")

    def __call__(self, spec):
        _, n_mels, n_steps = spec.shape
        F = math.ceil(n_mels * self.max_mask_pct) # rounding up in case of small %
        T = math.ceil(n_steps * self.max_mask_pct)
        fill = 0 if self.replace_with_zero else spec.mean()
        for i in range(0, self.freq_masks):
            f = random.randint(0, F)
            f0 = random.randint(0, n_mels-f)
            spec[0][f0:f0+f] = fill
        for i in range(0, self.time_masks):
            t = random.randint(0, T)
            t0 = random.randint(0, n_steps-t)
            spec[0][:,t0:t0+t] = fill
        return spec


masker = SpecAugment(freq_masks=2,time_masks=2,max_mask_pct=0.1)
show_spectro(masker(speccer(ll.train[0][0])))

# ### Using the transforms

# Now let's create the transforms with the params we want, and rebuild our label lists using them. 
#
# Note that now the items in the final `LabelList` won't be tuples anymore, they'll just be tensors. This is convenient for actually using the data, but it means you can't really go back and listen to your audio anymore. We can probably find a way around this, but let's press on for now.

# +
pad_3sec = PadOrTrim(3000)
shifter = SignalShift()
speccer = Spectrogrammer(n_fft=1024, n_mels=64, top_db=80)
masker = SpecAugment(freq_masks=2, time_masks=2, max_mask_pct=0.1)

tfms = [ToCuda(), shifter, pad_3sec, speccer, masker]

al = AudioList.from_files(path, tfms=tfms)
sd = SplitData.split_by_func(al, partial(random_splitter, p_valid=0.2))
ll = label_by_func(sd, speaker_labeler, proc_y=CategoryProcessor())
# -

show_spectro(ll.train[4][0])

# ## Databunch

# Now we've got our beautifully transformed tensors, let's add them into a databunch, so we can feed a model easily.
#
# We can use our `get_dls` func which we defined in `03_minibatch_training`, but let's use the to_databunch func we defined in `08_data_block` instead, it's much nicer.

# +
bs=64

c_in = ll.train[0][0].shape[0]
c_out = len(uniqueify(ll.train.y))
# -

data = ll.to_databunch(bs,c_in=c_in,c_out=c_out)

# Check the dataloader's batching functionality.

x,y = next(iter(data.train_dl))


# Here we're overriding `10_augmentation`'s `show_batch` method to take a "shower" function so we can give it a spectrogram shower.

#export
def show_batch(x, c=4, r=None, figsize=None, shower=show_image):
    n = len(x)
    if r is None: r = int(math.ceil(n/c))
    if figsize is None: figsize=(c*3,r*3)
    fig,axes = plt.subplots(r,c, figsize=figsize)
    for xi,ax in zip(x,axes.flat): shower(xi, ax)


show_spec_batch = partial(show_batch, c=4, r=2, figsize=None, 
                          shower=partial(show_spectro, with_shape=False))

show_spec_batch(x)

# Looking good.

# ## Training

# Go for gold! As a proof of concept, let's use the *pièce de résistance* learner builder with the hyperparameters from Lesson 11 `11_train_imagenette`.

opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)
loss_func = LabelSmoothingCrossEntropy()
lr = 1e-2
pct_start = 0.5
phases = create_phases(pct_start)
sched_lr  = combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5))
sched_mom = combine_scheds(phases, cos_1cycle_anneal(0.95,0.85, 0.95))
cbscheds = [ParamScheduler('lr', sched_lr), 
            ParamScheduler('mom', sched_mom)]

learn = cnn_learner(xresnet34, data, loss_func, opt_func)

learn.fit(5, cbs=cbscheds)

# ## Demo - all at once

# This is all the code it takes to do it end-to-end (not counting the `#export` cells above).

# +
# dsid = "ST-AEDS-20180100_1-OS"
# data_url = f'http://www.openslr.org/resources/45/{dsid}' # actual URL has .tgz extension but untar_data doesn't like that
# path = Path.home() / Path(f".fastai/data/{dsid}/")
# datasets.untar_data(data_url, dest=path)

# pad_3sec = PadOrTrim(3000)
# shifter = SignalShift()
# speccer = Spectrogrammer(n_fft=1024, n_mels=64, top_db=80)
# masker = SpecAugment(freq_masks=2, time_masks=2, max_mask_pct=0.1)

# tfms = [ToCuda(), shifter, pad_3sec, speccer, masker]
# al = AudioList.from_files(path, tfms=tfms)

# sd = SplitData.split_by_func(al, partial(random_splitter, p_valid=0.2))

# label_pat = r'/([mf]\d+)_'
# speaker_labeler = partial(re_labeler, pat=label_pat)
# ll = label_by_func(sd, speaker_labeler, proc_y=CategoryProcessor())

# bs=64
# c_in = ll.train[0][0].shape[0]
# c_out = len(uniqueify(ll.train.y))

# data = ll.to_databunch(bs,c_in=c_in,c_out=c_out)

# opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)
# loss_func = LabelSmoothingCrossEntropy()
# lr = 1e-2
# pct_start = 0.5
# phases = create_phases(pct_start)
# sched_lr  = combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5))
# sched_mom = combine_scheds(phases, cos_1cycle_anneal(0.95,0.85, 0.95))
# cbscheds = [ParamScheduler('lr', sched_lr), 
#             ParamScheduler('mom', sched_mom)]

# learn = cnn_learner(xresnet34, data, loss_func, opt_func)
# learn.fit(5, cbs=cbscheds)
# -

# ## Fin

nb_auto_export()


