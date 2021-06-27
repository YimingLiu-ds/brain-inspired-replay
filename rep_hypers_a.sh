#!/usr/bin/env bash
tsp -S 4
for lam in 1e1 1e2 1e3 1e4
do
	tsp hare run --rm -d --workdir /app -v "$(pwd)":/app --user $(id -u):$(id -g) --gpus device=3 jlm67/project python3 main_cl.py --experiment=splitMNIST --scenario=class --replay=generative --brain-inspired --si --repulsion --kl-js=js --use-rep-f --rep-f=6 --lamda-rep=$lam --tuning --seed=11
done