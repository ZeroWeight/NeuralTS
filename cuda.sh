source activate pytorch_latest_p37

python3 train.py --encoding multi --learner neural --style ts --nu 0.000001 --lamdba 0.01 --inv diag --dataset mushroom
python3 train.py --encoding multi --learner neural --style ts --nu 0.000001 --lamdba 0.01 --inv diag --dataset MagicTelescope
python3 train.py --encoding multi --learner neural --style ts --nu 0.000001 --lamdba 0.01 --inv diag --dataset covertype
python3 train.py --encoding multi --learner neural --style ts --nu 0.000001 --lamdba 0.01 --inv diag --dataset shuttle
python3 train.py --encoding multi --learner neural --style ts --nu 0.000001 --lamdba 0.01 --inv diag --dataset adult
python3 train.py --encoding multi --learner neural --style ts --nu 0.000001 --lamdba 0.01 --inv diag --dataset mnist

python3 train.py --encoding multi --learner neural --style ts --nu 0.000001 --lamdba 0.01 --inv diag --dataset mushroom
python3 train.py --encoding multi --learner neural --style ts --nu 0.000001 --lamdba 0.01 --inv diag --dataset MagicTelescope
python3 train.py --encoding multi --learner neural --style ts --nu 0.000001 --lamdba 0.01 --inv diag --dataset covertype
python3 train.py --encoding multi --learner neural --style ts --nu 0.000001 --lamdba 0.01 --inv diag --dataset shuttle
python3 train.py --encoding multi --learner neural --style ts --nu 0.000001 --lamdba 0.01 --inv diag --dataset adult
python3 train.py --encoding multi --learner neural --style ts --nu 0.000001 --lamdba 0.01 --inv diag --dataset mnist
