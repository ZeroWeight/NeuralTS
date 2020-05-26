# Kernel TS

# Mushroom, Multi: TBD: nu: 10 ~ 100 should < 0.1
python3 train.py --encoding multi --learner linear --style ts --nu 1 --lamdba 1 --dataset mushroom
python3 train.py --encoding multi --learner linear --style ucb --nu 1 --lamdba 1 --dataset mushroom
python3 train.py --encoding multi --learner kernel --style ts --nu 10 --lamdba 1 --dataset mushroom
python3 train.py --encoding multi --learner kernel --style ucb --nu 10 --lamdba 1 --dataset mushroom
python3 train.py --encoding multi --learner neural --style ts --nu 0.1 --lamdba 0.001 --dataset mushroom
python3 train.py --encoding multi --learner neural --style ucb --nu 0.1 --lamdba 0.001 --dataset mushroom
python3 train.py --encoding multi --learner diag --style ts --nu 0.1 --lamdba 0.001 --dataset mushroom
python3 train.py --encoding multi --learner diag --style ucb --nu 0.1 --lamdba 0.001 --dataset mushroom

# Covertype
python3 train.py --encoding multi --learner linear --style ucb --nu 1 --lamdba 1 --dataset covertype
python3 train.py --encoding multi --learner linear --style ts --nu 1 --lamdba 1 --dataset covertype
python3 train.py --encoding multi --learner kernel --style ucb --nu 10 --lamdba 10 --dataset covertype
python3 train.py --encoding multi --learner kernel --style ts --nu 1 --lamdba 1 --dataset covertype

# shuttle
python3 train.py --encoding multi --learner linear --style ucb --nu 1 --lamdba 1 --dataset shuttle
python3 train.py --encoding multi --learner linear --style ts --nu 0.1 --lamdba 1 --dataset shuttle
python3 train.py --encoding multi --learner kernel --style ucb --nu 1 --lamdba 1 --dataset shuttle
python3 train.py --encoding multi --learner kernel --style ts --nu 0.1 --lamdba 1 --dataset shuttle

# Magic Telescope
python3 train.py --encoding multi --learner linear --style ucb --nu 1 --lamdba 1 --dataset MagicTelescope
python3 train.py --encoding multi --learner linear --style ts --nu 1 --lamdba 1 --dataset MagicTelescope
python3 train.py --encoding multi --learner kernel --style ucb --nu 1 --lamdba 1 --dataset MagicTelescope
python3 train.py --encoding multi --learner kernel --style ts --nu 1 --lamdba 1 --dataset MagicTelescope

# MNIST (need a very small perturbation!!!)
python3 train.py --encoding multi --learner kernel --style ucb --nu 10 --lamdba 10 --dataset mnist
python3 train.py --encoding multi --learner kernel --style ts --nu 1 --lamdba 1 --dataset mnist
# Working One-hot
python3 train.py --encoding onehot --learner neural --style ts --nu 0.1 --lamdba 0.001 --dataset mushroom
python3 train.py --encoding onehot --learner neural --style ucb --nu 0.1 --lamdba 0.001 --dataset mushroom
python3 train.py --encoding onehot --learner diag --style ts --nu 0.1 --lamdba 0.001 --dataset mushroom
python3 train.py --encoding onehot --learner diag --style ucb --nu 0.1 --lamdba 0.001 --dataset mushroom



# Mushroom, one-hot, not good
# python3 train.py --encoding onehot --learner kernel --style ts --nu 1 --lamdba 1 --dataset mushroom
# python3 train.py --encoding onehot --learner kernel --style ucb --nu 1 --lamdba 1 --dataset mushroom