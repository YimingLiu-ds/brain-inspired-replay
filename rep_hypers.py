# -*- coding: utf-8 -*-
"""
Created on Thu Jun 24 16:25:19 2021

@author: jackm
"""

#! python

import subprocess
import sys
print (sys.version)

# Whether to use KL or JS divergence...
kl_js_list = ['kl', 'js']

# Selectrion factors...
f_list = [1.25, 1.5, 2, 2.5, 3, 3.5, 4, 5]

for kl_js in kl_js_list:
    for f in f_list:
        subprocess.run(['tsp', '-S 8', 'hare', 'run', '--rm', '--workdir /app', '-v "$(pwd)":/app', \
         '--user $(id', '-u):$(id', '-g)', '--gpus', '\'\"device=3,4\"\'', 'jlm67/project', \
         'python3', 'main_cl.py', '--experiment=splitMNIST', '--scenario=class', \
         '--replay=generative', '--brain-inspired', '--si', '--repulsion', '--kl-js={}'.format(kl_js), \
         '--use-rep-f', '--rep-f={}'.format(f), 'iters=10'])

#['tsp', '-S 8', 'hare', 'run', '--rm', '--workdir /app', '-v "$(pwd)":/app', \
#         '--user $(id', '-u):$(id', '-g)', '--gpus', '\'\"device=3,4\"\'', 'jlm67/project', \
#         'python', 'main_cl.py', '--experiment=splitMNIST', '--scenario=class', \
#         '--replay=generative', '--brain-inspired', '--si', '--repulsion', '--kl-js={}'.format(kl_js), \
#         '--use-rep-f', '--rep-f={}'.format(f), 'iters=10']      