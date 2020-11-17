# Linear UCB / TS
python3 train.py --encoding multi --learner linear --style ucb --nu 1 --lamdba 1 --inv full --dataset mushroom 
python3 train.py --encoding multi --learner linear --style ts --nu 0.1 --lamdba 1 --inv full --dataset mushroom
python3 train.py --encoding multi --learner linear --style ucb --nu 1 --lamdba 1 --inv full --dataset shuttle
python3 train.py --encoding multi --learner linear --style ts --nu 0.1 --lamdba 1 --inv full --dataset shuttle
python3 train.py --encoding multi --learner linear --style ucb --nu 1 --lamdba 1 --inv full --dataset adult
python3 train.py --encoding multi --learner linear --style ts --nu 0.1 --lamdba 1 --inv full --dataset adult
python3 train.py --encoding multi --learner linear --style ucb --nu 1 --lamdba 1 --inv full --dataset MagicTelescope
python3 train.py --encoding multi --learner linear --style ts --nu 1 --lamdba 1 --inv full --dataset MagicTelescope
python3 train.py --encoding multi --learner linear --style ucb --nu 1 --lamdba 1 --inv full --dataset covertype
python3 train.py --encoding multi --learner linear --style ts --nu 1 --lamdba 1 --inv full --dataset covertype
python3 train.py --encoding multi --learner linear --style ucb --nu 10 --lamdba 1 --inv full --dataset letter
python3 train.py --encoding multi --learner linear --style ts --nu 0.1 --lamdba 1 --inv full --dataset letter
python3 train.py --encoding multi --learner linear --style ucb --nu 0.001 --lamdba 0.001 --inv diag --dataset mnist
python3 train.py --encoding multi --learner linear --style ts --nu 0.001 --lamdba 0.001 --inv diag --dataset mnist
python3 train.py --encoding multi --learner linear --style ucb --nu 0.001 --lamdba 0.001 --inv diag --dataset isolet
python3 train.py --encoding multi --learner linear --style ts --nu 0.001 --lamdba 0.001 --inv diag --dataset isolet

# Neural UCB / TS
python3 train.py --encoding multi --learner neural --style ucb --nu 0.00001 --lamdba 0.00001 --inv diag --dataset shuttle
python3 train.py --encoding multi --learner neural --style ts --nu 0.00001 --lamdba 0.00001 --inv diag --dataset shuttle
python3 train.py --encoding multi --learner neural --style ucb --nu 0.00001 --lamdba 0.00001 --inv diag --dataset mushroom
python3 train.py --encoding multi --learner neural --style ts --nu 0.00001 --lamdba 0.00001 --inv diag --dataset mushroom
python3 train.py --encoding multi --learner neural --style ucb --nu 0.00001 --lamdba 0.00001 --inv diag --dataset mnist
python3 train.py --encoding multi --learner neural --style ts --nu 0.00001 --lamdba 0.00001 --inv diag --dataset mnist

python3 train.py --encoding multi --learner kernel --style ucb --nu 0.1 --lamdba 1 --inv full --dataset mnist
python3 train.py --encoding multi --learner kernel --style ts --nu 0.01 --lamdba 1 --inv full --dataset mnist
python3 train.py --encoding multi --learner kernel --style ucb --nu 0.1 --lamdba 1 --inv full --dataset mushroom
python3 train.py --encoding multi --learner kernel --style ts --nu 0.01 --lamdba 1 --inv full --dataset mushroom
