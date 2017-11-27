import numpy as np

from ..utils import *

''' L1 or L2 loss '''
gamma_space = lambda: 1

'''weighted loss - beta weight '''
beta_space = lambda: np.random.randint(100)

''' dense/lstm network size '''
units_space = lambda: 1024

''' regularizers '''
#l2_space      = lambda: np.random.choice([10 ** np.random.uniform(-1, -4), 0])
#dropout_space = lambda: np.random.uniform(0, 0.3)

l2_space      = lambda: 0
#dropout_space = lambda: np.random.uniform(0, 0.3)
dropout_space = lambda: 0

''' learning rate settings'''
#lr_space          = lambda: 10 ** np.random.uniform(-3.8, -4.1)
lr_space          = lambda: 2e-4
lr_modifier_space = lambda: NoModifier()
#lr_modifier_space = lambda: Scheduler()

#__scheduler_interval = lambda: 1
#__scheduler_factor   = lambda: 0.4


''' optimizer '''
optimizer_space       = lambda: Adam()
#__adam_epsilon_space  = lambda: np.random.choice([1, 0.1, 1e-8])
__adam_epsilon_space  = lambda: 1e-8

space = make_hyperparam_space__(globals())
desc  = make_descriptor(space)
