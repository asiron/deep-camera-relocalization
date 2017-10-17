import numpy as np

r = lambda f: lambda: round(f(), 2)

gamma_space   = lambda: 1
beta_space    = lambda: np.exp(np.random.uniform(6, 6))
l_rate_space  = lambda: 10 ** np.random.uniform(-5, -3.7)
#dropout_space = lambda: np.random.uniform(0.1, 0.5)
dropout_space = lambda: 0.5
#l2_regu_space = lambda: np.random.uniform(0.1, 0.4)
l2_regu_space = lambda: 0.01

decay_space   = lambda: np.random.choice([50])

space = {
  'gamma'   : gamma_space, 
  'beta'    : r(beta_space), 
  'l_rate'  : l_rate_space, 
  'dropout' : r(dropout_space),
  'l2_regu' : r(l2_regu_space),
  'decay'   : decay_space
}

desc = ('L={gamma},decay={decay},beta={beta:.1f},lr={l_rate:.2e},'
  + 'dropout={dropout:.2f},l2_regu={l2_regu:.2f}')

def make_l_rate_scheduler(base_l_rate, epochs):
  return lambda e: base_l_rate * (0.7 ** (e//epochs))
