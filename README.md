# Code for Neural Thompson Sampling

> @article{zhang2020neural,  
>  title={Neural Thompson Sampling},  
>  author={Zhang, Weitong and Zhou, Dongruo and Li, Lihong and Gu, Quanquan},  
>  journal={arXiv preprint arXiv:2010.00827},  
>  year={2020}  
>}  

## Dependencies and Installation

- Our code requires `PyTorch`, `CUDA` and `scikit-learn` for basic requirements
- See `requirements.txt` for more details and use `pip3 install -r requirements.txt --user` to install the packages.

## Code structure

- `train.py`: entry point of the program
- `data_multi.py`: data preprocessor to generate the disjoint feature encoding
- `learner_linear.py`: Linear Thompson Sampling / UCB
- `learner_kernel.py`: Kernel Thompson Sampling / UCB (cuda required!)
- `learner_diag.py`: Neural Thomoson Sampling (ours) / Neural UCB
- `neural_boost.py`: BootstrapNN and eps-greedy
- For other code, they are only for sanity check purpose and you do not need to care them.

## How to run

- First before running any experiments, check that you have a directory called `record` to save the `pkl` files
- To run the experiments described in our paper, simply type `sh ./run.sh`
- For feature encoding, always select `--encoding multi` since we do not report other encoding in our paper.
- `--dataset [adult|covertype|MagicTelescope|MNIST|mushroom|shuttle]` set the data set provided in our paper.
- For learner and how to get the inverse, we have ``--learner [linear|kernel] --inv full`` for linear TS / UCB and ``--learner neural --inv diag`` for Neural TS (this paper) and Neural UCB.
- For TS / UCB, set `--style [ts|ucb]`.
- `--lamdba`, `--nu` is the \lambda and \nu in the TS / UCB method, notice that it is `--lamdba` instead of `--lambda` and for Neural Networks, `--lamdba` is of 1 / m scale of \lambda in paper.
- `--hidden`: hidden layer size.
- `--p`, `--q` is the parameter for BoostrapNN, specially, setting `--p 1 --q 1` will lead to eps-greedy.
- `--delay` is the delay for delay update, for dynamic online update, --delay is set to default 1
- For any other combinations of hyparameters, try ``` python3 train.py -h```, but we will not provide good choice of parameter for other experiments beyound the ones claimed in our paper.

## Contract Information

Please contact [Weitong Zhang](mailto:weightzero[at]g[dot]ucla[dot]edu) if you find any difficulty running this program, or finding any issue with the results. You can also start a new issue on this repo but I will check the issue less often than email.
