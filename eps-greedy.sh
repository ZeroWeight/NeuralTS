source activate pytorch_latest_p37
for((n=0;n<2;n++))
do
	python3 train.py --encoding multi --learner boost --p 1 --q 1 --dataset mushroom --hidden 100
	python3 train.py --encoding multi --learner boost --p 1 --q 1 --dataset mnist --hidden 100
	python3 train.py --encoding multi --learner boost --p 1 --q 1 --dataset shuttle --hidden 100
	python3 train.py --encoding multi --learner boost --p 1 --q 1 --dataset covertype --hidden 100
	python3 train.py --encoding multi --learner boost --p 1 --q 1 --dataset MagicTelescope --hidden 100
	python3 train.py --encoding multi --learner boost --p 1 --q 1 --dataset adult --hidden 100

	python3 train.py --encoding multi --learner boost --p 0.8 --q 10 --dataset mushroom --hidden 100
	python3 train.py --encoding multi --learner boost --p 0.8 --q 10 --dataset mnist --hidden 100
	python3 train.py --encoding multi --learner boost --p 0.8 --q 10 --dataset shuttle --hidden 100
	python3 train.py --encoding multi --learner boost --p 0.8 --q 10 --dataset covertype --hidden 100
	python3 train.py --encoding multi --learner boost --p 0.8 --q 10 --dataset MagicTelescope --hidden 100
	python3 train.py --encoding multi --learner boost --p 0.8 --q 10 --dataset adult --hidden 100
done
