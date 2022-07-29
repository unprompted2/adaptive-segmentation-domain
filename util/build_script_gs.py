# This code builds the grid search scripts that will be launched on HPC

import os
import argparse

import numpy as np

from neuralnets.util.io import mkdir

EPFL = 'EPF