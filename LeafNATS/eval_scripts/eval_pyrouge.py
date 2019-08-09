'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import re
import os
import argparse
import shutil
from os import urandom

from pyrouge import Rouge155
r = Rouge155()

def run_pyrouge(args):
    '''
    Use pyrouge to evaluate scores.
    You need to install pyrouge, which can be installed from our tools.
    '''
    curr_key = urandom(5).hex()
    rouge_path = os.path.join('..', 'nats_results', curr_key)
    if os.path.exists(rouge_path):
        shutil.rmtree(rouge_path)
    os.makedirs(rouge_path)
    sys_smm_path = os.path.join(rouge_path, 'system_summaries')
    mod_smm_path = os.path.join(rouge_path, 'model_summaries')
    os.makedirs(sys_smm_path)
    os.makedirs(mod_smm_path)
    fp = open(os.path.join('..', 'nats_results', args.file_output), 'r')
    cnt = 1
    for line in fp:
        arr = re.split('<sec>', line[:-1])
        # reference
        rmm = re.split('<pad>|<s>|</s>', arr[1])
        rmm = list(filter(None, rmm))
        rmm = [' '.join(list(filter(None, re.split('\s', sen)))) for sen in rmm]
        rmm = list(filter(None, rmm))
        # generated
        smm = re.split('<stop>', arr[0])
        smm = list(filter(None, smm))
        smm = re.split('<pad>|<s>', smm[0])
        smm = list(filter(None, smm))
        smm = [sen.replace('</s>', '') for sen in smm if '</s>' in sen]
        smm = [' '.join(list(filter(None, re.split('\s', sen)))) for sen in smm]
        smm = list(filter(None, smm))
        fout = open(os.path.join(sys_smm_path, 'sum.'+str(cnt).zfill(5)+'.txt'), 'w')
        for sen in rmm:
            arr = re.split('\s', sen)
            arr = list(filter(None, arr))
            dstr = ' '.join(arr)
            fout.write(dstr+'\n')
        fout.close()
        fout = open(os.path.join(mod_smm_path, 'sum.A.'+str(cnt).zfill(5)+'.txt'), 'w')
        for sen in smm:
            arr = re.split('\s', sen)
            arr = list(filter(None, arr))
            dstr = ' '.join(arr)
            fout.write(dstr+'\n')
        fout.close()
        cnt += 1
    fp.close()

    path_to_rouge = os.path.abspath(os.path.join('..', 'nats_results'))
    r.system_dir = os.path.join(path_to_rouge, curr_key, 'system_summaries')
    r.model_dir = os.path.join(path_to_rouge, curr_key, 'model_summaries')
    r.system_filename_pattern = 'sum.(\d+).txt'
    r.model_filename_pattern = 'sum.[A-Z].#ID#.txt'

    output = r.convert_and_evaluate()
    print(output)
    fout = open(os.path.join('..', 'nats_results', 'rouge_'+curr_key+'_'+args.file_output), 'w')
    fout.write(output)
    fout.close()

    shutil.rmtree(rouge_path)
