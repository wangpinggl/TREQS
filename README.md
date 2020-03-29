# TREQS: A Translate-Edit Model for Question-to-SQL Query Generation

[![image](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![image](https://img.shields.io/pypi/l/ansicolortags.svg)](https://github.com/wangpinggl/TREQS/blob/master/LICENSE)
[![image](https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg)](https://github.com/wangpinggl/TREQS/graphs/contributors)
[![image](https://img.shields.io/badge/arXiv-1908.01839-red.svg?style=flat)](https://arxiv.org/abs/1908.01839)

- This repository is a pytorch implementation of the TREQS model for Question-to-SQL generation proposed in our WWW'20 paper:
[Text-to-SQL Generation for Question Answering on Electronic Medical Records](http://dmkd.cs.vt.edu/papers/WWW20.pdf). 

- In this work, we also created a large-scale dataset MIMICSQL for Question-to-SQL task in healthcare domain. The MIMICSQL dataset is provided in folder [mimicsql_data](https://github.com/wangpinggl/TREQS/tree/master/mimicsql_data) in this repository. More details about MIMICSQL dataset are provided below.

## Dataset
MIMICSQL is created based on the publicly available real-world [Medical Information Mart for Intensive Care III (MIMIC III)](https://mimic.physionet.org/gettingstarted/access/) dataset.  

- ```Database:``` We extracted five categories of information for 46,520 patients, including demographics, laboratory tests, diagnosis, procedures and prescriptions, and prepared a specific table for each category separately. These tables compose a relational patient database where tables are linked through patient ID and admission ID.

- ```Questions:``` MIMICSQL has two subsets, in which the first set is composed of template questions (machine generated), while the second consists of natural language questions (human annotated). Generally, each template question is rephrased as one natural language question. Recently, we add more natural language questions for a subset of template questions. Readers are refered to get more detailed information for question generation and basic statistics of MIMICSQL dataset.

- ```Example:``` Here we provide a data sample in MIMICSQL to illustrate the meaning of each element.


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
