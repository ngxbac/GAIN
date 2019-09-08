# flake8: noqa
from catalyst.dl import registry
from .experiment import Experiment
from .runner import ModelRunner as Runner
from models import *
from callbacks import *
from optimizers import *


# Register models
registry.Model(GAIN)
registry.Model(GCAM)

# Register callbacks
registry.Callback(GAINCriterionCallback)
registry.Callback(GAINSaveHeatmapCallback)
registry.Callback(GCAMSaveHeatmapCallback)
registry.Callback(GAINMaskCriterionCallback)

# Register criterions

# Register optimizers
registry.Optimizer(AdamW)
registry.Optimizer(Nadam)