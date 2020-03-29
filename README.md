# TREQS: A Translate-Edit Model for Question-to-SQL Query Generation

[![image](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![image](https://img.shields.io/pypi/l/ansicolortags.svg)](https://github.com/wangpinggl/TREQS/blob/master/LICENSE)
[![image](https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg)](https://github.com/wangpinggl/TREQS/graphs/contributors)
[![image](https://img.shields.io/badge/arXiv-1908.01839-red.svg?style=flat)](https://arxiv.org/abs/1908.01839)

- This repository is a pytorch implementation of the TREQS model for Question-to-SQL generation proposed in our WWW'20 paper:
[Text-to-SQL Generation for Question Answering on Electronic Medical Records](http://dmkd.cs.vt.edu/papers/WWW20.pdf),
[Ping Wang](https://github.com/wangpinggl),
[Tian Shi](https://github.com/tshi04), 
[Chandan K. Reddy](http://people.cs.vt.edu/~reddy/).

## Dataset
The MIMICSQL dataset is provided in folder 'mimicsql_data'.

## Usuage

- ```Training:``` python main.py 

- ```Validate:``` python main.py --task validate

- ```Test:``` python main.py --task test

## Citation

```
@inproceedings{wang2020treqs,
 title={Text-to-SQL Generation for Question Answering on Electronic Medical Records},
 author={Wang, Ping and Shi, Tian and Reddy, Chandan K.},
 booktitle={Proceedings of the 2020 World Wide Web Conference},
 year={2020}
}
```
