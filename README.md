# Text-to-SQL Generation for Question Answering on Electronic Medical Records

[![image](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![image](https://img.shields.io/pypi/l/ansicolortags.svg)](https://github.com/wangpinggl/TREQS/blob/master/LICENSE)
[![image](https://img.shields.io/badge/arXiv-1908.01839-red.svg?style=flat)](https://arxiv.org/abs/1908.01839)

- This repository is a pytorch implementation of the TREQS model for Question-to-SQL generation proposed in our WWW'20 paper:
[Text-to-SQL Generation for Question Answering on Electronic Medical Records](http://dmkd.cs.vt.edu/papers/WWW20.pdf). 

- In this work, we are also releasing a large-scale dataset MIMICSQL for Question-to-SQL generation task in healthcare domain. The MIMICSQL dataset is provided in folder [mimicsql_data](https://github.com/wangpinggl/TREQS/tree/master/mimicsql_data) in this repository. More details about MIMICSQL dataset are provided below.

- Links related to this work:
  - Paper: http://dmkd.cs.vt.edu/papers/WWW20.pdf
  - Dataset and codes: https://github.com/wangpinggl/TREQS
  - Slides: https://github.com/wangpinggl/TREQS/blob/master/TREQS.pdf
  - Presentation Video: [Video](https://drive.google.com/open?id=1tXRaobsz1BWUJpzV976pgox_46c8jkPE)
  
- Updates (06/2020) : We recently further improve the quality of human annotated questions and release a new version of natural language questions in `mimicsql_data/mimicsql_natual_v2`. The model performance on this new version of data can be found as follows.

## Citation
Ping Wang, Tian Shi, and Chandan K. Reddy. Text-to-SQL Generation for Question Answering on Electronic Medical Records. In Proceedings of The Web Conference 2020 (WWW’20), April 20–24, 2020, Taipei, Taiwan.

```
@inproceedings{wang2020text,
  title={Text-to-SQL Generation for Question Answering on Electronic Medical Records},
  author={Wang, Ping and Shi, Tian and Reddy, Chandan K},
  booktitle={Proceedings of The Web Conference 2020},
  pages={350--361},
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

- ```Questions:``` We do not enumerate all possible questions about MIMIC III dataset. MIMICSQL dataset is generated based on  the patient information related to 100 randomly sampled hospital admissions. MIMICSQL has two subsets, in which the first set is composed of template questions (machine generated), while the second consists of natural language questions (human annotated). Generally, each template question is rephrased as one natural language question. Readers are refered to read the [paper](http://dmkd.cs.vt.edu/papers/WWW20.pdf) get more detailed information for question generation and basic statistics of MIMICSQL dataset.

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
- `question_refine`: the machine genrated template question (in `mimicsql_template` folder) or natural language question (in `mimicsql_natural` folder) annotated by [Freelancers](https://www.freelancer.com/). You can make correspondence between template question and natural language questions using `key` of each data sample.
- `question_refine_tok`: the tokenized question. The content is ignored here. You can find details in the dataset.
- `sql`: the SQL query corresponding to the question.
- `sql_tok`: the tokenized SQL query. The content is ignored here. You can find details in the dataset.
- `format`: the logical format of SQL query, which is inspired by the logical format used in [WikiSQL](https://github.com/salesforce/WikiSQL). It includes the following sub-elements:
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

## Usage

- ```Training:``` python main.py 

- ```Validate:``` python main.py --task validate

- ```Test:``` python main.py --task test

## Evaluation

The codes for evaluation are provided in folder ```evaluation```. You can follow the following steps to evaluate the generated queries.

- Update the path to the MIMIC III data and generate MIMIC III database ```mimic.db``` with ```process_mimic_db.py```.

- Generate lookup table with ```build_lookup.ipynb```.

- Run overall evaluation or breakdown evaluation. If you plan to apply condition value recover technique, you need to run overall evaluation first (which will save the generated SQL queries with recovered condition values) before getting breakdown performance. Also, for evaluating on testing and development set, make sure to use the corresponding data file test.json and dev.json for testing and development sets, respectively.

## Results

Here we provide the results on the new version of natual language questions.

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th colspan="2">Overall Evaluation</th>
      <th colspan="6">Breakdown Evaluation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td></td><td>Acc_LF</td><td>Acc_EX</td><td>Agg_op</td><td>Agg_col</td><td>Table</td><td>Con_col+op</td><td>Con_val</td><td>Average</td>
    </tr>
    <tr>
      <td>Testing</td><td>0.482</td><td>0.611</td><td>0.993</td><td>0.970</td><td>0.954</td><td>0.857</td><td>0.630</td><td>0.881</td>
    </tr>
    <tr>
      <td>Testing+recover</td><td>0.547</td><td>0.690</td><td>0.992</td><td>0.969</td><td>0.953</td><td>0.863</td><td>0.729</td><td>0.901</td>
    </tr>
    <tr>
      <td>Development</td><td>0.432</td><td>0.636</td><td>0.997</td><td>0.988</td><td>0.956</td><td>0.845</td><td>0.524</td><td>0.862</td>
    </tr>
    <tr>
      <td>Development+recover</td><td>0.526</td><td>0.741</td><td>0.997</td><td>0.988</td><td>0.956</td><td>0.837</td><td>0.639</td><td>0.883</td>
    </tr>
  </tbody>
</table>

<!-- | Dataset              | Overall Evaluation || Breakdown Evaluation                               ||||||
|-|-|-|-|-|-|-|-|-|
|                      |Acc\_LF   | Acc\_EX |Agg\_op | Agg\_col | Table |Con\_col+op |Con\_val |Average|
|Testing               | 0.482    |0.611    |0.993   | 0.970    |0.954  |0.854       |0.630    |0.881  |
|Testing + recover     |0.547     |0.690    |0.992   | 0.969    |0.953  |0.863       |0.729    |0.901  |
|Development           | 0.432    |0.636    |0.997   | 0.988    |0.956  |0.845       |0.524    |0.862  |
|Development + recover | 0.526    |0.741    |0.997   | 0.988    |0.956  |0.837       |0.639    |0.883  |
 -->