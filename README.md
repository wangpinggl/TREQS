# Text-to-SQL Generation for Question Answering on Electronic Medical Records

[![image](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![image](https://img.shields.io/pypi/l/ansicolortags.svg)](https://github.com/wangpinggl/TREQS/blob/master/LICENSE)
[![image](https://img.shields.io/badge/arXiv-1908.01839-red.svg?style=flat)](https://arxiv.org/abs/1908.01839)

- This repository is a pytorch implementation of the TREQS model for Question-to-SQL generation proposed in our WWW'20 paper:
[Text-to-SQL Generation for Question Answering on Electronic Medical Records](http://dmkd.cs.vt.edu/papers/WWW20.pdf). 

- In this work, we are also releasing a large-scale dataset MIMICSQL for Question-to-SQL generation task in healthcare domain. The MIMICSQL dataset is provided in folder [mimicsql_data](https://github.com/wangpinggl/TREQS/tree/master/mimicsql_data) in this repository. More details about MIMICSQL dataset are provided below.

## Citation
Ping Wang, Tian Shi, and Chandan K. Reddy. Text-to-SQL Generation for Question Answering on Electronic Medical Records. In Proceedings of The Web Conference 2020 (WWW’20), April 20–24, 2020, Taipei, Taiwan.

```
@inproceedings{wang2020text,
 title={Text-to-SQL Generation for Question Answering on Electronic Medical Records},
 author={Wang, Ping and Shi, Tian and Reddy, Chandan K.},
 booktitle={Proceedings of The Web Conference 2020 (WWW'20)},
 year={2020}
}
```

## Dataset
MIMICSQL is created based on the publicly available real-world de-identified [Medical Information Mart for Intensive Care III (MIMIC III)](https://mimic.physionet.org/gettingstarted/access/) dataset. In order to generated more realistic questions, each patient is randomly assigned a synthetic name, which should not be used to identify any patients.

- ```Database:``` We extracted five categories of information for 46,520 patients, including demographics, laboratory tests, diagnosis, procedures and prescriptions, and prepared a specific table for each category separately. These tables compose a relational patient database where tables are linked through patient ID and admission ID. The numerical indexes of tables is `{'Demographic': 0, 'Diagnoses': 1, 'Procedures': 2, 'Prescriptions': 3, 'Lab': 4}`. The columns included in each table are as follows:
  - `Demographic`: `['hadm_id', ‘subject_id’, ‘name’, ‘marital status’, ‘age’, ‘date of birth’, ‘gender’, ‘language’, ‘religion’, ‘admission type’, ‘days of hospital stay’, ‘insurance’, ‘ethnicity’, ‘death status’, ‘admission location’, ‘discharge location’, ‘primary disease’, ‘date of death’, ‘year of birth’, ‘year of death’, ‘admission time’, ‘discharge time’, ‘admission year’]`
  - `Diagnoses`: `[‘subject_id’, ‘hadm_id’, ‘diagnoses icd9 code’, ‘diagnoses short title’, ‘diagnoses long title’]`
  - `Procedures`: `[‘subject_id’, ‘hadm_id’, ‘procedure icd9 code’, ‘procedure short title’, ‘procedure long title’]`
  - `Prescriptions`: `[‘subject_id’, ‘hadm_id’, ‘icustay_id’, ‘drug type’, ‘drug name’, ‘drug code’, ‘drug route’, ‘drug dose’]`
  - `Lab`: `[‘subject_id’, ‘hadm_id’, ‘itemid’, ‘lab test chart time’, ‘lab test abnormal status’, ‘lab test value’, ‘lab test name’, ‘lab test fluid’, ‘lab test category’]`

- ```Questions:``` MIMICSQL has two subsets, in which the first set is composed of template questions (machine generated), while the second consists of natural language questions (human annotated). Generally, each template question is rephrased as one natural language question. Readers are refered to read the [paper](http://dmkd.cs.vt.edu/papers/WWW20.pdf) get more detailed information for question generation and basic statistics of MIMICSQL dataset.

- ```Example:``` Here we provide a data sample in MIMICSQL to illustrate the meaning of each element.

```json
{
  "key": "a81dae5ff42498734e857c5b7dc46deb",
  "format": {
    "table": [
      0,
      2
    ],
    "cond": [
      [
        0,
        6,
        0,
        "F"
      ],
      [
        2,
        3,
        0,
        "Abdomen artery incision"
      ]
    ],
    "agg_col": [
      [
        0,
        0
      ]
    ],
    "sel": 1
  },
  "question_refine": "how many female patients underwent the procedure of abdomen artery incision?",
  "sql": "SELECT COUNT ( DISTINCT DEMOGRAPHIC.\"SUBJECT_ID\" ) FROM DEMOGRAPHIC INNER JOIN PROCEDURES on DEMOGRAPHIC.HADM_ID = PROCEDURES.HADM_ID WHERE DEMOGRAPHIC.\"GENDER\" = \"F\" AND PROCEDURES.\"SHORT_TITLE\" = \"Abdomen artery incision\"",
  "question_refine_tok": [],
  "sql_tok": []
}
```

The meaning of each elements are as follows:
- `key`: a unique ID of each data sample. You can make correspondence between the template question and natural language question using this ID.
- `question_refine`: the machine genrated template question (in `mimicsql_template` folder) or natural language question (in `mimicsql_natural` folder) annotated by [Freelancers](https://www.freelancer.com/). You can make correspondence between tmplate question and natural language questions using `key` of each data sample.
- `question_refine_tok`: the tokenized question. The content is ignored here. You can find details in the dataset.
- `sql`: the SQL query corresponding to the question.
- `sql_tok`: the tokenized SQL query. The content is ignored here. You can find details in the dataset.
- `format`: the logical format of SQL query, which included the following sub-elements:
  - `table`: a list of numerical index of tables that are related to the question.
  - `cond`: a list of `[talble_index, condition_column_index, condition_operation_index, condition_value]`, where:
    - `table_index`: the numerical index of table that is related to this condition column.
    - `condition_column_index`: the numerical index of column that is used for this condition.
    - `condition_operation_index`: the numerical index of condition operation. Here is the dictionary `{'=': 0, '>': 1, '<': 2, '>=': 3, '<=': 4}`.
    - `condition_value`: the value for the condition. 
  - `agg_col`: a list of `[talble_index, aggreation_column_index]`.
    - `table_index`: the numerical index of table that is related to this aggregation column.
    - `column_index`: the numerical index of column that is used for aggregation.
  - `sel`: the numerical index of aggregation operation used in the SQL query. Here is the dictionary `{'': 0, 'count': 1, 'max': 2, 'min': 3, 'avg': 4}`.

## Usuage

- ```Training:``` python main.py 

- ```Validate:``` python main.py --task validate

- ```Test:``` python main.py --task test
