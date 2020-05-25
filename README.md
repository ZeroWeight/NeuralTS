# Comparison of UCB and Thompson Sampling method

## Checklist
 - `OH`: One-hot
 - `MB`: Multi-Bandit
 - `SC`: Sanity-Check
### UCB

|Experiments|Linear|Kernel|Neural|DiagNeural|
|----|----|----|----|----|
|Linear|<ul><li>- [] SC</li><ul>|<ul><li>- [] SC</li><ul>|<ul><li>- [] SC</li><ul>|<ul><li>- [] SC</li><ul>|
|MNIST|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|Mushroom|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|Adult|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|Covertype|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|ISOLet|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|Letter|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|Magic|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|

### Thompson Sampling

|Experiments|Linear|Kernel|Neural|DiagNeural|
|----|----|----|----|----|
|Linear|<ul><li>- [] SC</li><ul>|<ul><li>- [] SC</li><ul>|<ul><li>- [] SC</li><ul>|<ul><li>- [] SC</li><ul>|
|MNIST|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|Mushroom|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|Adult|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|Covertype|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|ISOLet|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|Letter|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|Magic|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|

### $\epsilon$-greedy

|Experiments|Linear|Kernel|Neural|DiagNeural|
|----|----|----|----|----|
|Linear|<ul><li>- [] SC</li><ul>|<ul><li>- [] SC</li><ul>|<ul><li>- [] SC</li><ul>|<ul><li>- [] SC</li><ul>|
|MNIST|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|Mushroom|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|Adult|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|Covertype|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|ISOLet|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|Letter|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|
|Magic|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|<ul><li>- [] OH</li><li>- [] MB</li><ul>|