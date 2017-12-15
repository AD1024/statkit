__all__ = ['basic', 'regression', 'probability']


try:
	from . import basic
	from . import regression
	from . import probability
except ImportError:
	from statkit import basic
	from statkit import regression
	from statkit import probability
