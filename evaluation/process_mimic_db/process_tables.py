import os
import csv
import shutil
import pandas
import numpy as np
from datetime import datetime

from process_mimic_db.utils import *

def build_demographic_table(data_dir, out_dir, conn):
    print('Build demographic_table')
    pat_id2name = get_patient_name('process_mimic_db')
    pat_info = read_table(data_dir, 'PATIENTS.csv')
    adm_info = read_table(data_dir, 'ADMISSIONS.csv')
    print('-- Process PATIENTS')
    cnt = 0
    for itm in pat_info:
        cnt += 1
        show_progress(cnt, len(pat_info))
        itm['NAME'] = pat_id2name[itm['SUBJECT_ID']]
        
        dob = datetime.strptime(itm['DOB'], '%Y-%m-%d %H:%M:%S')
        itm['DOB_YEAR'] = str(dob.year)
        
        if len(itm['DOD']) > 0:
            dod = datetime.strptime(itm['DOD'], '%Y-%m-%d %H:%M:%S')
            itm['DOD_YEAR'] = str(dod.year)
        else:
            itm['DOD_YEAR'] = ''
            
    pat_dic = {ky['SUBJECT_ID']: ky for ky in pat_info}
    print()
    print('-- Process ADMISSIONS')
    cnt = 0
    for itm in adm_info:
        cnt += 1
        show_progress(cnt, len(adm_info))
        # patients.csv
        for ss in pat_dic[itm['SUBJECT_ID']]:
            if ss == 'ROW_ID' or ss == 'SUBJECT_ID':
                continue
            itm[ss] = pat_dic[itm['SUBJECT_ID']][ss]
        # admissions.csv
        admtime = datetime.strptime(itm['ADMITTIME'], '%Y-%m-%d %H:%M:%S')
        itm['ADMITYEAR'] = str(admtime.year)
        dctime = datetime.strptime(itm['DISCHTIME'], '%Y-%m-%d %H:%M:%S')
        itm['DAYS_STAY'] = str((dctime-admtime).days)
        itm['AGE'] = str(int(itm['ADMITYEAR'])-int(itm['DOB_YEAR']))
        if int(itm['AGE']) > 89:
            itm['AGE'] = str(89+int(itm['AGE'])-300)
    print()
    print('-- write table')
    header = [
        'SUBJECT_ID',
        'HADM_ID',
        'NAME',
        'MARITAL_STATUS',
        'AGE',
        'DOB',
        'GENDER',
        'LANGUAGE',
        'RELIGION',
        
        'ADMISSION_TYPE',
        'DAYS_STAY',
        'INSURANCE',
        'ETHNICITY',
        'EXPIRE_FLAG',
        'ADMISSION_LOCATION',
        'DISCHARGE_LOCATION',
        'DIAGNOSIS',
        
        'DOD',
        'DOB_YEAR',
        'DOD_YEAR',
        
        'ADMITTIME',
        'DISCHTIME',
        'ADMITYEAR'
    ]
            
    fout = open(os.path.join(out_dir,'DEMOGRAPHIC.csv'), 'w')    
    fout.write('\"'+'\",\"'.join(header)+'\"\n')
    for itm in adm_info:
        arr = []
        for wd in header:
            arr.append(itm[wd])
        fout.write('\"'+'\",\"'.join(arr)+'\"\n')
    fout.close()
    print('-- write sql')
    data = pandas.read_csv(
        os.path.join(out_dir,'DEMOGRAPHIC.csv'), 
        dtype={'HADM_ID': str, "DOD_YEAR": float, "SUBJECT_ID": str})
    data.to_sql('DEMOGRAPHIC', conn, if_exists='replace', index=False)

def build_diagnoses_table(data_dir, out_dir, conn):
    print('Build diagnoses_table')
    left = pandas.read_csv(os.path.join(data_dir, 'DIAGNOSES_ICD.csv'), dtype=str)
    right = pandas.read_csv(os.path.join(data_dir, 'D_ICD_DIAGNOSES.csv'), dtype=str)
    left = left.drop(columns=['ROW_ID', 'SEQ_NUM'])
    right = right.drop(columns=['ROW_ID'])
    out = pandas.merge(left, right, on='ICD9_CODE')
    out = out.sort_values(by='HADM_ID')
    print('-- write table')
    out.to_csv(os.path.join(out_dir, 'DIAGNOSES.csv'), sep=',', index=False)
    print('-- write sql')
    out.to_sql('DIAGNOSES', conn, if_exists='replace', index=False)
    
def build_procedures_table(data_dir, out_dir, conn):
    print('Build procedures_table')
    left = pandas.read_csv(os.path.join(data_dir, 'PROCEDURES_ICD.csv'), dtype=str)
    right = pandas.read_csv(os.path.join(data_dir, 'D_ICD_PROCEDURES.csv'), dtype=str)
    left = left.drop(columns=['ROW_ID', 'SEQ_NUM'])
    right = right.drop(columns=['ROW_ID'])
    out = pandas.merge(left, right, on='ICD9_CODE')
    out = out.sort_values(by='HADM_ID')
    print('-- write table')
    out.to_csv(os.path.join(out_dir, 'PROCEDURES.csv'), sep=',', index=False)
    print('-- write sql')
    out.to_sql('PROCEDURES', conn, if_exists='replace', index=False) 
    
def build_prescriptions_table(data_dir, out_dir, conn):
    print('Build prescriptions_table')
    data = pandas.read_csv(os.path.join(data_dir, 'PRESCRIPTIONS.csv'), dtype=str)
    data = data.drop(columns=['ROW_ID', 'GSN', 'DRUG_NAME_POE', 
                              'DRUG_NAME_GENERIC', 'NDC', 'PROD_STRENGTH', 
                              'FORM_VAL_DISP', 'FORM_UNIT_DISP', 
                              'STARTDATE', 'ENDDATE'])
    data = data.dropna(subset=['DOSE_VAL_RX', 'DOSE_UNIT_RX'])
    data['DRUG_DOSE'] = data[['DOSE_VAL_RX', 'DOSE_UNIT_RX']].apply(lambda x: ''.join(x), axis=1)
    data = data.drop(columns=['DOSE_VAL_RX', 'DOSE_UNIT_RX'])
    print('-- write table')
    data.to_csv(os.path.join(out_dir, 'PRESCRIPTIONS.csv'), sep=',', index=False)
    print('-- write sql')
    data.to_sql('PRESCRIPTIONS', conn, if_exists='replace', index=False) 
    
def build_lab_table(data_dir, out_dir, conn):
    print('Build lab_table')
    cnt = 0
    show_progress(cnt, 4)
    left = pandas.read_csv(os.path.join(data_dir, 'LABEVENTS.csv'), dtype=str)
    cnt += 1
    show_progress(cnt, 4)
    right = pandas.read_csv(os.path.join(data_dir, 'D_LABITEMS.csv'), dtype=str)
    cnt += 1
    show_progress(cnt, 4)
    left = left.dropna(subset=['HADM_ID', 'VALUE', 'VALUEUOM'])
    left = left.drop(columns=['ROW_ID', 'VALUENUM'])
    left['VALUE_UNIT'] = left[['VALUE', 'VALUEUOM']].apply(lambda x: ''.join(x), axis=1)
    left = left.drop(columns=['VALUE', 'VALUEUOM'])
    right = right.drop(columns=['ROW_ID', 'LOINC_CODE'])
    cnt += 1
    show_progress(cnt, 4)
    out = pandas.merge(left, right, on='ITEMID')
    cnt += 1
    show_progress(cnt, 4)
    print()
    print('-- write table')
    out.to_csv(os.path.join(out_dir, 'LAB.csv'), sep=',', index=False)
    print('-- write sql')
    out.to_sql('LAB', conn, if_exists='replace', index=False)
    