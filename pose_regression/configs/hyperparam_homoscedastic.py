import numpy as np

#gamma_space   = lambda: np.random.randint(1, high=3)
gamma_space   = lambda: 2
beta_space    = lambda: 151.8
l_rate_space  = lambda: 5.12e-07
dropout_space = lambda: 0.28
l2_regu_space = lambda: 0.39
decay_space   = lambda: 240

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

desc = ('decay={decay},beta={beta:.1f},lr={l_rate:.2e},'
      + 'dropout={dropout:.2f},l2_regu={l2_regu:.2f}')

def make_l_rate_scheduler(base_l_rate, epochs):
  return lambda e: base_l_rate * (0.9 ** (e//epochs))
