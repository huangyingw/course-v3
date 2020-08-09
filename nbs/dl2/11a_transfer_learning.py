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

# +
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline
# -

#export
from exp.nb_11 import *

# ## Serializing the model

# [Jump_to lesson 12 video](https://course.fast.ai/videos/?lesson=12&t=2920)

path = datasets.untar_data(datasets.URLs.IMAGEWOOF_160)

# +
size = 128
bs = 64

tfms = [make_rgb, RandomResizedCrop(size, scale=(0.35,1)), np_to_float, PilRandomFlip()]
val_tfms = [make_rgb, CenterCrop(size), np_to_float]
il = ImageList.from_files(path, tfms=tfms)
sd = SplitData.split_by_func(il, partial(grandparent_splitter, valid_name='val'))
ll = label_by_func(sd, parent_labeler, proc_y=CategoryProcessor())
ll.valid.x.tfms = val_tfms
data = ll.to_databunch(bs, c_in=3, c_out=10, num_workers=8)
# -

len(il)

loss_func = LabelSmoothingCrossEntropy()
opt_func = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)

learn = cnn_learner(xresnet18, data, loss_func, opt_func, norm=norm_imagenette)


def sched_1cycle(lr, pct_start=0.3, mom_start=0.95, mom_mid=0.85, mom_end=0.95):
    phases = create_phases(pct_start)
    sched_lr  = combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5))
    sched_mom = combine_scheds(phases, cos_1cycle_anneal(mom_start, mom_mid, mom_end))
    return [ParamScheduler('lr', sched_lr),
            ParamScheduler('mom', sched_mom)]


lr = 3e-3
pct_start = 0.5
cbsched = sched_1cycle(lr, pct_start)

learn.fit(40, cbsched)

st = learn.model.state_dict()

type(st)

', '.join(st.keys())

st['10.bias']

mdl_path = path/'models'
mdl_path.mkdir(exist_ok=True)

# It's also possible to save the whole model, including the architecture, but it gets quite fiddly and we don't recommend it. Instead, just save the parameters, and recreate the model directly.

torch.save(st, mdl_path/'iw5')

# ## Pets

# [Jump_to lesson 12 video](https://course.fast.ai/videos/?lesson=12&t=3127)

pets = datasets.untar_data(datasets.URLs.PETS)

pets.ls()

pets_path = pets/'images'

il = ImageList.from_files(pets_path, tfms=tfms)

il


#export
def random_splitter(fn, p_valid): return random.random() < p_valid


random.seed(42)

sd = SplitData.split_by_func(il, partial(random_splitter, p_valid=0.1))

sd

n = il.items[0].name; n

re.findall(r'^(.*)_\d+.jpg$', n)[0]


def pet_labeler(fn): return re.findall(r'^(.*)_\d+.jpg$', fn.name)[0]


proc = CategoryProcessor()

ll = label_by_func(sd, pet_labeler, proc_y=proc)

', '.join(proc.vocab)

ll.valid.x.tfms = val_tfms

c_out = len(proc.vocab)

data = ll.to_databunch(bs, c_in=3, c_out=c_out, num_workers=8)

learn = cnn_learner(xresnet18, data, loss_func, opt_func, norm=norm_imagenette)

learn.fit(5, cbsched)

# ## Custom head

# [Jump_to lesson 12 video](https://course.fast.ai/videos/?lesson=12&t=3265)

learn = cnn_learner(xresnet18, data, loss_func, opt_func, c_out=10, norm=norm_imagenette)

st = torch.load(mdl_path/'iw5')

m = learn.model

m.load_state_dict(st)

cut = next(i for i,o in enumerate(m.children()) if isinstance(o,nn.AdaptiveAvgPool2d))
m_cut = m[:cut]

xb,yb = get_batch(data.valid_dl, learn)

pred = m_cut(xb)

pred.shape

ni = pred.shape[1]


#export
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=1):
        super().__init__()
        self.output_size = sz
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)


# +
nh = 40

m_new = nn.Sequential(
    m_cut, AdaptiveConcatPool2d(), Flatten(),
    nn.Linear(ni*2, data.c_out))
# -

learn.model = m_new

learn.fit(5, cbsched)


# ## adapt_model and gradual unfreezing

# [Jump_to lesson 12 video](https://course.fast.ai/videos/?lesson=12&t=3483)

def adapt_model(learn, data):
    cut = next(i for i,o in enumerate(learn.model.children())
               if isinstance(o,nn.AdaptiveAvgPool2d))
    m_cut = learn.model[:cut]
    xb,yb = get_batch(data.valid_dl, learn)
    pred = m_cut(xb)
    ni = pred.shape[1]
    m_new = nn.Sequential(
        m_cut, AdaptiveConcatPool2d(), Flatten(),
        nn.Linear(ni*2, data.c_out))
    learn.model = m_new


learn = cnn_learner(xresnet18, data, loss_func, opt_func, c_out=10, norm=norm_imagenette)
learn.model.load_state_dict(torch.load(mdl_path/'iw5'))

adapt_model(learn, data)

for p in learn.model[0].parameters(): p.requires_grad_(False)

learn.fit(3, sched_1cycle(1e-2, 0.5))

for p in learn.model[0].parameters(): p.requires_grad_(True)

learn.fit(5, cbsched, reset_opt=True)

# ## Batch norm transfer

# [Jump_to lesson 12 video](https://course.fast.ai/videos/?lesson=12&t=3567)

learn = cnn_learner(xresnet18, data, loss_func, opt_func, c_out=10, norm=norm_imagenette)
learn.model.load_state_dict(torch.load(mdl_path/'iw5'))
adapt_model(learn, data)


# +
def apply_mod(m, f):
    f(m)
    for l in m.children(): apply_mod(l, f)

def set_grad(m, b):
    if isinstance(m, (nn.Linear,nn.BatchNorm2d)): return
    if hasattr(m, 'weight'):
        for p in m.parameters(): p.requires_grad_(b)


# -

apply_mod(learn.model, partial(set_grad, b=False))

learn.fit(3, sched_1cycle(1e-2, 0.5))

apply_mod(learn.model, partial(set_grad, b=True))

learn.fit(5, cbsched, reset_opt=True)

# Pytorch already has an `apply` method we can use:

learn.model.apply(partial(set_grad, b=False));

# ## Discriminative LR and param groups

# [Jump_to lesson 12 video](https://course.fast.ai/videos/?lesson=12&t=3799)

learn = cnn_learner(xresnet18, data, loss_func, opt_func, c_out=10, norm=norm_imagenette)

learn.model.load_state_dict(torch.load(mdl_path/'iw5'))
adapt_model(learn, data)


def bn_splitter(m):
    def _bn_splitter(l, g1, g2):
        if isinstance(l, nn.BatchNorm2d): g2 += l.parameters()
        elif hasattr(l, 'weight'): g1 += l.parameters()
        for ll in l.children(): _bn_splitter(ll, g1, g2)
        
    g1,g2 = [],[]
    _bn_splitter(m[0], g1, g2)
    
    g2 += m[1:].parameters()
    return g1,g2


a,b = bn_splitter(learn.model)

test_eq(len(a)+len(b), len(list(m.parameters())))

Learner.ALL_CBS

#export
from types import SimpleNamespace
cb_types = SimpleNamespace(**{o:o for o in Learner.ALL_CBS})

cb_types.after_backward


#export
class DebugCallback(Callback):
    _order = 999
    def __init__(self, cb_name, f=None): self.cb_name,self.f = cb_name,f
    def __call__(self, cb_name):
        if cb_name==self.cb_name:
            if self.f: self.f(self.run)
            else:      set_trace()


#export
def sched_1cycle(lrs, pct_start=0.3, mom_start=0.95, mom_mid=0.85, mom_end=0.95):
    phases = create_phases(pct_start)
    sched_lr  = [combine_scheds(phases, cos_1cycle_anneal(lr/10., lr, lr/1e5))
                 for lr in lrs]
    sched_mom = combine_scheds(phases, cos_1cycle_anneal(mom_start, mom_mid, mom_end))
    return [ParamScheduler('lr', sched_lr),
            ParamScheduler('mom', sched_mom)]


disc_lr_sched = sched_1cycle([0,3e-2], 0.5)

# +
learn = cnn_learner(xresnet18, data, loss_func, opt_func,
                    c_out=10, norm=norm_imagenette, splitter=bn_splitter)

learn.model.load_state_dict(torch.load(mdl_path/'iw5'))
adapt_model(learn, data)


# +
def _print_det(o): 
    print (len(o.opt.param_groups), o.opt.hypers)
    raise CancelTrainException()

learn.fit(1, disc_lr_sched + [DebugCallback(cb_types.after_batch, _print_det)])
# -

learn.fit(3, disc_lr_sched)

disc_lr_sched = sched_1cycle([1e-3,1e-2], 0.3)

learn.fit(5, disc_lr_sched)

# ## Export

!./notebook2script.py 11a_transfer_learning.ipynb


