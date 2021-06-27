#!/usr/bin/env bash
tsp -S 4
for lam in 1e5 1e6 1e7 1e8
do
	tsp hare run --rm -d --workdir /app -v "$(pwd)":/app --user $(id -u):$(id -g) --gpus device=5 jlm67/project python3 main_cl.py --experiment=splitMNIST --scenario=class --replay=generative --brain-inspired --si --repulsion --kl-js=js --use-rep-f --rep-f=6 --lamda-rep=$lam --tuning --seed=11
done