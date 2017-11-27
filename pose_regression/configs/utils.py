import functools

def add_name(cls):
  def __format__(self, format_spec):
    return self.__class__.__name__.lower()
  cls.__format__ = __format__
  return cls

def make_hyperparam_space__(spaces):
  def fix_name(name):
    return name.replace('_space', '')

  return {fix_name(name): func \
    for (name, func) in spaces.items() \
    if callable(func) and name.endswith('_space')}

def make_descriptor(space):
  return  ','.join(['{0}={{{0}}}'.format(name) \
    for name in space.keys() \
    if not name.startswith('__')])

@add_name
class Reducer(object):

  def __call__(self, **hyperparams):
    from keras.callbacks import ReduceLROnPlateau
    factor   = hyperparams.get('__reducer_factor', 0.8)
    patience = hyperparams.get('__reducer_patience', 2)
    cooldown = hyperparams.get('__reducer_cooldown', 1)
    return ReduceLROnPlateau(
      monitor='val_loss',
      factor=factor,
      patience=patience,
      min_lr=1e-6,
      cooldown=cooldown,
      verbose=True,
      epsilon=1e-4)

@add_name
class Scheduler(object):

  def __call__(self, **hyperparams):
    from keras.callbacks import LearningRateScheduler
    base_lr  = hyperparams['lr']
    interval = hyperparams.get('__scheduler_interval', 5)
    factor   = hyperparams.get('__scheduler_factor', 0.6)
    scheduler = lambda e: base_lr * (factor ** (e//interval))
    return LearningRateScheduler(scheduler)

@add_name
class NoModifier(object):

  def __call__(self, **hyperparams):
    return None

@add_name
class Adam(object):

  def __call__(self, **hyperparams):
    from keras.optimizers import Adam
    lr = hyperparams['lr']
    epsilon = hyperparams['__adam_epsilon']
    return Adam(lr=lr, epsilon=epsilon)

@add_name
class SGD(object):
  
  def __call__(self, **hyperparams):
    from keras.optimizers import SGD
    lr = hyperparams['lr']
    nesterov = hyperparams['__sgd_nesterov']
    return SGD(lr=lr, momentum=0.9, nesterov=nesterov)
