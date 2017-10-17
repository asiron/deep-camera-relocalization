import numpy as np

#gamma_space   = lambda: np.random.randint(1, high=3)
gamma_space   = lambda: 2
beta_space    = lambda: np.exp(np.random.uniform(4, 5.5))
l_rate_space  = lambda: 10 ** np.random.uniform(-6, -4)
dropout_space = lambda: np.random.uniform(0.1, 0.5)
l2_regu_space = lambda: np.random.uniform(0.1, 0.3)

space = {
  'gamma'   : gamma_space, 
  'beta'    : beta_space, 
  'l_rate'  : l_rate_space, 
  'dropout' : dropout_space,
  'l2_regu' : l2_regu_space
}

desc = ('L{gamma},beta={beta:.1f},lr={l_rate:.2e},'
      + 'dropout={dropout:.2f},l2_regu={l2_regu:.2f}')

def make_l_rate_scheduler(base_l_rate):
  return lambda e: base_l_rate * (0.9 ** (e//150))
