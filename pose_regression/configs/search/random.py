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
lr_space          = lambda: 10 ** np.random.uniform([-3.8, -4.1])
lr_modifier_space = lambda: np.random.choice([Reducer(), NoModifier(), Scheduler()])

__reducer_factor_space   = lambda: 0.8
__reducer_patience_space = lambda: 2
__reducer_cooldown_space = lambda: 1

__scheduler_interval_space = lambda: 5
__scheduler_factor_space   = lambda: 0.6


''' optimizer '''
optimizer_space = lambda: np.random.choice([Adam(), SGD()])

__sgd_nesterov  = lambda: True
__adam_epsilon_space  = lambda: np.random.choice([1, 0.1, 1e-8])


space = make_hyperparam_space__(globals())
desc  = make_descriptor(space)
