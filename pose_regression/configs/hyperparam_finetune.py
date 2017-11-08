import numpy as np

#gamma_space   = lambda: np.random.choice([1, 2])
gamma_space   = lambda: 1

beta_space = lambda: np.exp(np.random.uniform(4.7874898999, 6.5808989999))
#beta_space    = lambda: 16.444646771097055
#beta_space    = lambda: 32

#lstm_units_space = lambda: int(2 ** np.random.uniform(5, 10))
#lstm_units_space = lambda: int(2 ** np.random.uniform(6.5, 8.5))
#lstm_units_space = lambda: np.random.choice([7, 10, 16])
lstm_units_space = lambda: 104

#l_rate_space = lambda: 10 ** np.random.uniform(-3.8, -4.1)
l_rate_space = lambda: 1.27e-4

#dropout_space = lambda: np.random.uniform(0.01, 0.7)
dropout_space = lambda: 0.0001

#l2_regu_space = lambda: np.random.uniform(0.01, 0.4) 
l2_regu_space = lambda: 0.0001

#decay_space   = lambda: np.random.choice([10, 20, 30])
decay_space   = lambda: 5

#build_space = lambda: np.random.choice(['standard', 'spatial'])
#r_act_space = lambda: np.random.choice(['hard_sigmoid', 'tanh'])

build_space = lambda: 'standard'
r_act_space = lambda: 'tanh'


space = {
  'gamma'   : gamma_space,
  'beta'    : beta_space, 
  'l_rate'  : l_rate_space, 
  'dropout' : dropout_space,
  'l2_regu' : l2_regu_space,
  'decay'   : decay_space,
  'lstm_units' : lstm_units_space,
  'build' : build_space,
  'r_act' : r_act_space
}

desc = ('gamma={gamma},decay={decay},beta={beta:.1f},l_rate={l_rate:.2e},'
      + 'dropout={dropout:.3f},l2_regu={l2_regu:.3f},lstm_units={lstm_units},build={build},r_act={r_act}')

def make_l_rate_scheduler(base_l_rate, epochs):
  return lambda e: base_l_rate * (0.60 ** (e//epochs))
