# Code for Neural Thompson Sampling submitted to NIPS2020

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

## Results (also in Appendix A in our paper)
| |Adult|Covertype|Magic|MNIST|Mushroom|Shuttle|
|-|-----|---------|-----|-----|--------|-------|
|Round\#|10000|10000|10000|10000|8124|10000|
|Input Dim|2 * 15|2 * 55|2 * 12|10 * 784|2 * 23|7 * 9|
|Random|5000|5000|5000|9000|4062|8571|
|Linear UCB|2078.0 \pm 47.1|3220.4 \pm 59.0|2616.2 \pm 29.6|2544.0 \pm 235.4|569.6 \pm 18.1|956.5 \pm 22.9|
|Linear TS|2118.1 \pm 41.7|3385.4 \pm 72.1|2605.2 \pm 33.3|2781.4 \pm 338.3|625.4 \pm 60.7|1045.6 \pm 53.8|
|Kernel UCB|**2060.5\pm 20.1**|3547.2 \pm 103.9|2405.1 \pm 85.6|3399.5 \pm 258.4|182.9 \pm 32.9|**182.2\pm 24.3**|
|Kernel TS|2110.2 \pm 88.3|3693.0 \pm 123.6|2415.9 \pm 47.5|3385.1 \pm 401.0|278.9 \pm 37.6|270.2 \pm 63.8|
|BootstrapNN|2095.5 \pm 44.8|3060.2 \pm 66.1|2267.2 \pm 30.8|1776.6 \pm 380.9|130.5 \pm 9.9|210.6 \pm 25.2|
|\epsilon-greedy|2328.5 \pm 50.4|3334.2 \pm 72.6|2381.8 \pm 37.3|1893.2 \pm 93.7|323.2 \pm 32.5|682.0 \pm 79.8|
|Neural UCB|2102.8 \pm 33.1|**3058.5\pm 39.3**|**2074.0\pm 43.6**|1531.0 \pm 268.4|84.5 \pm 23.7|209.6 \pm 105.8|
|Neural TS (ours)|2088.2 \pm 69.8|3069.2 \pm 73.1|2088.4 \pm 54.6|**1522.8\pm 194.6**|**83.1 \pm 37.4**|242.5 \pm 206.7|

$\pm$ here is the plus-minus sign here.
