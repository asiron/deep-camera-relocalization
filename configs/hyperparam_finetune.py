import numpy as np

#gamma_space   = lambda: np.random.randint(1, high=3)
gamma_space   = lambda: np.random.choice([1, 2])
#beta_space    = lambda: np.exp(np.random.uniform(4.7874898999, 6.5808989999))
beta_space    = lambda: 321.9

#l_rate_space  = lambda: 
l_rate_space = lambda: 10 ** np.random.uniform(-3, -5.2)
#dropout_space = lambda: np.random.uniform(0.1, 0.7)
dropout_space = lambda: np.random.uniform(0.1, 0.5)
l2_regu_space = lambda: np.random.uniform(0.01, 0.001)
#l2_regu_space = lambda: 0
decay_space   = lambda: np.random.choice([7, 15])

# #gamma_space   = lambda: np.random.randint(1, high=3)
# gamma_space   = lambda: 2
# beta_space    = lambda: 47.5
# #l_rate_space  = lambda: 0.0001
# l_rate_space = lambda: 10 ** np.random.uniform(-4, -6)
# #dropout_space = lambda: np.random.uniform(0.1, 0.7)
# dropout_space = lambda: 0.34
# l2_regu_space = lambda: 0.14
# #l2_regu_space = lambda: 0
# decay_space   = lambda: np.random.choice([10])

#beta=47.5,lr=9.74e-06,dropout=0.34,l2_regu=0.14

space = {
  'gamma'   : gamma_space, 
  'beta'    : beta_space, 
  'l_rate'  : l_rate_space, 
  'dropout' : dropout_space,
  'l2_regu' : l2_regu_space,
  'decay'   : decay_space
}

# desc = ('L{gamma},beta={beta:.1f},lr={l_rate:.2e},'
#       + 'dropout={dropout:.2f},l2_regu={l2_regu:.2f}')

desc = ('L{gamma},decay={decay},beta={beta:.1f},lr={l_rate:.2e},'
      + 'dropout={dropout:.3f},l2_regu={l2_regu:.3f}')

#desc = ('decay={decay},lr={l_rate:.2e},'
#      + 'dropout={dropout:.2f},l2_regu={l2_regu:.2f}')

def make_l_rate_scheduler(base_l_rate, epochs):
  return lambda e: base_l_rate * (0.9 ** (e//epochs))
