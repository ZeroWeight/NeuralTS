for((n=0;n<15;n++))
do
	nohup python3 train.py --encoding multi --learner linear --style ucb --nu 1 --lamdba 1 --inv full --dataset mnist &
done
