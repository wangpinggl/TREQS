# TREQS
A Translate-Edit Model for Question to SQL Query Generation

[![image](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![image](https://img.shields.io/pypi/l/ansicolortags.svg)](https://github.com/wangpinggl/TREQS/blob/master/LICENSE)
[![image](https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg)](https://github.com/wangpinggl/TREQS/graphs/contributors)
[![image](https://img.shields.io/badge/arXiv-1805.09461-red.svg?style=flat)](https://arxiv.org/abs/1908.01839)

#### Our new implementation is available at https://github.com/wangpinggl/TREQS. Please check it.
- This repository is a pytorch implementation of the Translate-Edit Model for Question to SQL Query Generation proposed in the following paper:


[A Translate-Edit Model for Natural Language Question to SQL Query Generation on Multi-relational Healthcare Data](https://arxiv.org/abs/1908.01839)

[Ping Wang](https://github.com/wangpinggl)
[Tian Shi](https://github.com/tshi04), 
[Chandan K. Reddy](http://people.cs.vt.edu/~reddy/)

## Dataset
The MIMICSQL dataset will be released soon.

## Usuage

- ```Training:``` python main.py 

- ```Validate:``` python main.py --task validate

- ```Test:``` python main.py --task test

If you use the code, please kindly consider citing our paper:

```
    @article{wang2019treqs,
     title={A Translate-Edit Model for Natural Language Question to SQL Query Generation on Multi-relational Healthcare Data},
     author={Wang, Ping and Shi, Tian and Reddy, Chandan K.},
     journal={arXiv preprint arXiv:1908.01839},
     year={2019}
    }
```
