#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 21:58:08 2022

@author: hill103

this script configure the environment variables for GLRM
"""



"""
overwrite the print function with the default set to flush = True
"""
import sys
def print(*objects, sep=' ', end='\n', file=sys.stdout, flush=True):
    import builtins
    builtins.print(*objects, sep=sep, end=end, file=file, flush=flush)



"""
version info
"""
cur_version = '1.0.5'



# The [is-docker package for npm](https://github.com/sindresorhus/is-docker/blob/master/index.js) suggests a robust approach to determine if it's running within a docker container
import os
def is_docker():
    path = '/proc/self/cgroup'
    return (
        os.path.exists('/.dockerenv') or
        os.path.isfile(path) and any('docker' in line for line in open(path))
    )



"""
define the input folder to store all input files
define the output path to store all result files
"""
if is_docker():
    # for Docker image
    input_path = r'/data'  
    output_path = r'/data'
else:
    input_path = ''
    output_path = os.getcwd()



"""
define a small value to avoid divided by 0 or log(0)
"""
min_val = 1e-12



"""
define a small value to make sure theta (w) > 0
"""
min_theta = 1e-9



"""
define a small value to make sure sigma^2 > 0
"""
min_sigma2 = 1e-2



"""
define the integration range and increment to calculate the heavy-tail probabilities
"""
N_z = 1000
gamma = 4e-3



"""
define number of digits used for round floats in fast computing the heavy-tail probabilities
"""
mu_digits = 10
sigma2_digits = 6



"""
define eps in optmization for theta and sigma2 (default 1e-08)
"""
theta_eps = 1e-8
sigma2_eps = 1e-8



"""
configuration for reproducible results in keras + TensorFlow
"""
# Seed value
#seed_value = 1154

# 1. Set 'PYTHONHASHSEED' environment variable at a fixed value (it must be set before Python running)
#import os
#os.environ['PYTHONHASHSEED'] = str(seed_value)

# tf.keras.utils.set_random_seed can set all random seeds for the program (Python, NumPy, and TensorFlow).

# 2. Set 'python' built-in pseudo-random generator at a fixed value
#import random
#random.seed(seed_value)

# 3. Set 'numpy' pseudo-random generator at a fixed value
#import numpy as np
#np.random.seed(seed_value)

# 4. Set 'tensorflow' pseudo-random generator at a fixed value
#import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.random.set_seed(seed_value)

# 5. Configure a new global `tensorflow` session
# skip setting here, set it later after get the user specified CPU cores
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)