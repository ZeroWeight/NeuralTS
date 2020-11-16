for ((n=0;n<16;n++))
do
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
done
