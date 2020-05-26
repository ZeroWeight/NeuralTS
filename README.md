# Comparison of UCB and Thompson Sampling method

## Checklist
 - `OH`: One-hot
 - `MB`: Multi-Bandit
 - `SC`: Sanity-Check
### UCB / TS

|Experiments|Linear|Kernel|Neural|DiagNeural|
|----|----|----|----|----|
|Linear|<ul><li>- [x] SC</li><ul>|<ul><li>- [x] SC</li><ul>|<ul><li>- [x] SC</li><ul>|<ul><li>- [x] SC</li><ul>|
|MNIST|Imp|<ul><li>- [ ] OH</li><li>- [x] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [x] MB</li><ul>|
|Mushroom|<ul><li>- [x] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [x] MB</li><ul>|<ul><li>- [x] OH</li><li>- [x] MB</li><ul>|<ul><li>- [x] OH</li><li>- [x] MB</li><ul>|
|Adult|<ul><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|
|Covertype|<ul>- [x] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [x] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|
|ISOLet|<ul><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|
|Letter|<ul><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|
|Magic|<ul><li>- [x] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [x] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|
|Shuttle|<ul><li>- [x] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [x] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|

### $\epsilon$-greedy

|Experiments|Linear|Kernel|Neural|DiagNeural|
|----|----|----|----|----|
|Linear|<ul><li>- [x] SC</li><ul>|<ul><li>- [x] SC</li><ul>|<ul><li>- [x] SC</li><ul>|<ul><li>- [x] SC</li><ul>|
|MNIST|<ul><li>- [x] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|
|Mushroom|<ul><li>- [x] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|
|Adult|<ul><li>- [x] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|
|Covertype|<ul><li>- [x] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|
|ISOLet|<ul><li>- [x] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|
|Letter|<ul><li>- [x] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|
|Magic|<ul><li>- [x] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|
|Shuttle|<ul><li>- [x] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|<ul><li>- [ ] OH</li><li>- [ ] MB</li><ul>|