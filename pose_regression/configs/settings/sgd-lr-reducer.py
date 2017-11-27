import numpy as np

from ..utils import *

''' L1 or L2 loss '''
gamma_space = lambda: 1

'''weighted loss - beta weight '''
beta_space = lambda: 300

''' dense/lstm network size '''
units_space = lambda: 256

''' regularizers '''
l2_space      = lambda: 0
dropout_space = lambda: 0

''' learning rate settings '''
lr_space          = lambda: 2e-4
lr_modifier_space = lambda: Reducer()
__reducer_factor_space   = lambda: 0.8
__reducer_patience_space = lambda: 2
__reducer_cooldown_space = lambda: 1


''' optimizer '''
optimizer_space      = lambda: SGD()
__sgd_nesterov_space = lambda: True


space = make_hyperparam_space__(globals())
desc  = make_descriptor(space)
