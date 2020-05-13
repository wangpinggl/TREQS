import os
import csv
import shutil
import sqlite3
from datetime import datetime

from process_mimic_db.utils import *
from process_mimic_db.process_tables import *

# Specify the path to the downloaded MIMIC III data
data_dir = PATH_TO_MIMIC_DATA
# Path to the generated mimic.db. No need to update.
out_dir = 'mimic_db'

# Generate five tables and the database with all admissions
# if os.path.exists(out_dir):
#     shutil.rmtree(out_dir)
# os.mkdir(out_dir)
conn = sqlite3.connect(os.path.join(out_dir, 'mimic_all.db'))
build_demographic_table(data_dir, out_dir, conn)
build_diagnoses_table(data_dir, out_dir, conn)
build_procedures_table(data_dir, out_dir, conn)
build_prescriptions_table(data_dir, out_dir, conn)
build_lab_table(data_dir, out_dir, conn)

'''
1. We did not emumerate all possible questions about MIMIC III.
MIMICSQL data is generated based on the patient information 
related to 100 randomly selected admissions.
2. The following codes are used for sampling the admissions 
from the large database. 
3. The parameter 'random_state=0' in line 41 will provide you 
the same set of sampled admissions and the same database as we used.
'''

print('Begin sampling ...')
# DEMOGRAPHIC
print('Processing DEMOGRAPHIC')
conn = sqlite3.connect(os.path.join(out_dir, 'mimic.db'))
data_demo = pandas.read_csv(os.path.join(out_dir, "DEMOGRAPHIC.csv"))
data_demo_sample = data_demo.sample(100, random_state=0)
data_demo_sample.to_sql('DEMOGRAPHIC', conn, if_exists='replace', index=False)
sampled_id = data_demo_sample['HADM_ID'].values

# DIAGNOSES
print('Processing DIAGNOSES')
data_input = pandas.read_csv(os.path.join(out_dir, "DIAGNOSES.csv"))
data_filter = []
cnt = 0
for itm in sampled_id:
    msg = 'HADM_ID=='+str(itm)
    data_filter.append(data_input.query(msg))
    cnt += 1
    show_progress(cnt, len(sampled_id))
data_out = pandas.concat(data_filter, ignore_index=True)
data_out.to_sql('DIAGNOSES', conn, if_exists='replace', index=False)

# PROCEDURES
print('Processing PROCEDURES')
data_input = pandas.read_csv(os.path.join(out_dir, "PROCEDURES.csv"))
data_filter = []
cnt = 0
for itm in sampled_id:
    msg = 'HADM_ID=='+str(itm)
    data_filter.append(data_input.query(msg))
    cnt += 1
    show_progress(cnt, len(sampled_id))
data_out = pandas.concat(data_filter, ignore_index=True)
data_out.to_sql('PROCEDURES', conn, if_exists='replace', index=False)

# PRESCRIPTIONS
print('Processing PRESCRIPTIONS')
data_input = pandas.read_csv(os.path.join(out_dir, "PRESCRIPTIONS.csv"))
data_filter = []
cnt = 0
for itm in sampled_id:
    msg = 'HADM_ID=='+str(itm)
    data_filter.append(data_input.query(msg))
    cnt += 1
    show_progress(cnt, len(sampled_id))
data_out = pandas.concat(data_filter, ignore_index=True)
data_out.to_sql('PRESCRIPTIONS', conn, if_exists='replace', index=False)

# LAB
print('Processing LAB')
data_input = pandas.read_csv(os.path.join(out_dir, "LAB.csv"))
data_filter = []
cnt = 0
for itm in sampled_id:
    msg = 'HADM_ID=='+str(itm)
    data_filter.append(data_input.query(msg))
    cnt += 1
    show_progress(cnt, len(sampled_id))
data_out = pandas.concat(data_filter, ignore_index=True)
data_out.to_sql('LAB', conn, if_exists='replace', index=False)
print('Done!')