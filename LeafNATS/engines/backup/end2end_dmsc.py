'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import re
import os
import json
import shutil
import glob
import time
import numpy as np
from pprint import pprint

import torch
from torch.autograd import Variable

from LeafNATS.data.utils import create_batch_memory
from LeafNATS.utils.utils import show_progress

class End2EndBase(object):
    '''
    End2End training for document level multi-aspect sentiment classification.
    Possibly extend to other classification tasks.
    Light weight. Data should be relevatively small.
    '''
    def __init__(self, args=None):
        '''
        Initialize
        '''
        self.args = args
        self.base_models = {}
        self.train_models = {}
        self.batch_data = {}
        self.train_data = []
        self.val_data = []
        self.test_data = []
        self.vis_data = []
        
        self.pred_data = []
        self.true_data = []
        
        self.global_steps = 0
        
    def build_vocabulary(self):
        '''
        vocabulary
        '''
        raise NotImplementedError
    
    def build_models(self):
        '''
        Models
        -- self.base_models: models that will not be trained.
              Format: {'name1': model1, 'name2': model2}
        -- self.train_models: models that will be trained.
              Format: {'name1': model1, 'name2': model2}
        '''
        raise NotImplementedError
        
    def init_base_model_params(self):
        '''
        Initialize base model parameters.
        self.base_models.
        '''
        raise NotImplementedError
    
    def init_train_model_params(self):
        '''
        Initialize train model parameters.
        self.train_models.
        for testing, visualization, app.
        '''
        raise NotImplementedError
        
    def build_pipelines(self):
        '''
        Pipelines and loss here.
        '''
        raise NotImplementedError
        
    def build_optimizer(self, params):
        '''
        define optimizer
        '''
        raise NotImplementedError
        
    def build_scheduler(self, optimizer):
        '''
        define optimizer learning rate scheduler.
        '''
        raise NotImplementedError
        
    def build_batch(self, batch_):
        '''
        process batch data.
        '''
        raise NotImplementedError
        
    def test_worker(self):
        '''
        Testing and Evaluation.
        '''
        raise NotImplementedError
        
    def visualization_worker(self, batch_id, vis_dir):
        '''
        Visualization Attention Weights.
        '''
        raise NotImplementedError
        
    def run_evaluation(self):
        '''
        Evaluation.
        '''
        raise NotImplementedError
    
    def train(self):
        '''
        training here.
        Don't overwrite.
        '''
        self.build_vocabulary()
        self.build_models()
        print(self.base_models)
        print(self.train_models)
        if len(self.base_models) > 0:
            self.init_base_model_params()
        # here it is necessary to put list. Instead of directly append. 
        for model_name in self.train_models:
            try:
                params += list(self.train_models[model_name].parameters())
            except:
                params = list(self.train_models[model_name].parameters())
        if self.args.train_base_model:
            for model_name in self.base_models:
                try:
                    params += list(self.base_models[model_name].parameters())
                except:
                    params = list(self.base_models[model_name].parameters())
        # define optimizer
        optimizer = self.build_optimizer(params)
        try:
            scheduler = self.build_scheduler(optimizer)
        except:
            pass
        # load checkpoint
        cc_model = 0
        out_dir = os.path.join('..', 'nats_results')
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        if self.args.continue_training:
            model_para_files = glob.glob(os.path.join(out_dir, '*.model'))
            if len(model_para_files) > 0:
                uf_model = []
                for fl_ in model_para_files:
                    arr = re.split('\/', fl_)[-1]
                    arr = re.split('\_|\.', arr)
                    if arr not in uf_model:
                        uf_model.append(int(arr[-2]))
                cc_model = sorted(uf_model)[-1]
                try:
                    print("Try *_{}.model".format(cc_model))
                    for model_name in self.train_models:
                        fl_ = os.path.join(
                            out_dir, model_name+'_'+str(cc_model)+'.model')
                        self.train_models[model_name].load_state_dict(
                            torch.load(fl_, map_location=lambda storage, loc: storage))
                except:
                    cc_model = sorted(uf_model)[-2]
                    print("Try *_{}.model".format(cc_model))
                    for model_name in self.train_models:
                        fl_ = os.path.join(
                            out_dir, model_name+'_'+str(cc_model)+'.model')
                        self.train_models[model_name].load_state_dict(
                            torch.load(fl_, map_location=lambda storage, loc: storage))
                print('Continue training with *_{}.model'.format(cc_model))
        else:
            shutil.rmtree(out_dir)
            os.mkdir(out_dir)
            
        self.val_data = create_batch_memory(
            path_=self.args.data_dir,
            file_=self.args.file_val,
            is_shuffle=False,
            batch_size=self.args.batch_size,
            is_lower=self.args.is_lower
        )
        self.test_data = create_batch_memory(
            path_=self.args.data_dir,
            file_=self.args.file_test,
            is_shuffle=False,
            batch_size=self.args.batch_size,
            is_lower=self.args.is_lower
        )
        # train models
        if cc_model > 0:
            cc_model -= 1
        for epoch in range(cc_model, self.args.n_epoch):
            '''
            Train
            '''
            for model_name in self.base_models: 
                self.base_models[model_name].train()
            for model_name in self.train_models: 
                self.train_models[model_name].train()
            print('====================================')
            print('Training Epoch: {}'.format(epoch+1))
            self.train_data = create_batch_memory(
                path_=self.args.data_dir,
                file_=self.args.file_train,
                is_shuffle=True,
                batch_size=self.args.batch_size,
                is_lower=self.args.is_lower
            )
            n_batch = len(self.train_data)
            print('The number of batches (training): {}'.format(n_batch))
            self.global_steps = max(0, epoch) * n_batch
            try:
                scheduler.step()
            except:
                pass
            if self.args.debug:
                n_batch = 1
            loss_arr = []    
            for batch_id in range(n_batch):
                self.global_steps += 1
                
                self.build_batch(self.train_data[batch_id])
                loss = self.build_pipelines()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, self.args.grad_clip)
                optimizer.step()

                if batch_id%self.args.checkpoint == 0:
                    for model_name in self.train_models:                    
                        fmodel = open(os.path.join(
                            out_dir, model_name+'_'+str(epoch+1)+'.model'), 'wb')
                        torch.save(self.train_models[model_name].state_dict(), fmodel)
                        fmodel.close()
                show_progress(batch_id+1, n_batch)
                loss_arr.append(loss.data.cpu().numpy())
            print()
            # write models
            print('Training Loss = {}.'.format(np.average(loss_arr)))
            for model_name in self.train_models:
                fmodel = open(os.path.join(
                    out_dir, model_name+'_'+str(epoch+1)+'.model'), 'wb')
                torch.save(self.train_models[model_name].state_dict(), fmodel)
                fmodel.close()
        
            for model_name in self.base_models: 
                self.base_models[model_name].eval()
            for model_name in self.train_models: 
                self.train_models[model_name].eval()
            with torch.no_grad():
                '''
                Validate
                '''
                print('Begin Validation')
                n_batch = len(self.val_data)
                print('The number of batches (validation): {}'.format(n_batch))
                self.pred_data = []
                self.true_data = []
                if self.args.debug:
                    n_batch = 1
                for batch_id in range(n_batch):

                    self.build_batch(self.val_data[batch_id])
                    ratePred, rateTrue = self.test_worker()
                    
                    self.pred_data += ratePred
                    self.true_data += rateTrue

                    show_progress(batch_id+1, n_batch)
                print()
                self.pred_data = np.array(self.pred_data).astype(int)
                np.savetxt(
                    os.path.join('..', 'nats_results', 'validate_pred_{}.txt'.format(epoch+1)), 
                    self.pred_data, fmt='%d')
    
                self.true_data = np.array(self.true_data).astype(int)
                np.savetxt(
                    os.path.join('..', 'nats_results', 'validate_true_{}.txt'.format(epoch+1)), 
                    self.true_data, fmt='%d')
            
                self.run_evaluation()
                '''
                Testing
                '''
                print('Begin Testing')
                n_batch = len(self.test_data)
                print('The number of batches (testing): {}'.format(n_batch))
                self.pred_data = []
                self.true_data = []
                if self.args.debug:
                    n_batch = 1
                for batch_id in range(n_batch):

                    self.build_batch(self.test_data[batch_id])
                    ratePred, rateTrue = self.test_worker()
                    
                    self.pred_data += ratePred
                    self.true_data += rateTrue

                    show_progress(batch_id+1, n_batch)
                print()
                self.pred_data = np.array(self.pred_data).astype(int)
                np.savetxt(
                    os.path.join('..', 'nats_results', 'test_pred_{}.txt'.format(epoch+1)), 
                    self.pred_data, fmt='%d')
    
                self.true_data = np.array(self.true_data).astype(int)
                np.savetxt(
                    os.path.join('..', 'nats_results', 'test_true_{}.txt'.format(epoch+1)), 
                    self.true_data, fmt='%d')
            
                self.run_evaluation()
    
    def visualization(self):
        '''
        Visualization
        '''
        self.build_vocabulary()
        self.build_models()
        print(self.base_models)
        print(self.train_models)
        if len(self.base_models) > 0:
            self.init_base_model_params()
        if len(self.train_models) > 0:
            self.init_train_model_params()
            
        self.vis_data = create_batch_memory(
            path_=self.args.data_dir,
            file_=self.args.file_vis,
            is_shuffle=False,
            batch_size=self.args.batch_size,
            is_lower=self.args.is_lower
        )
        
        vis_dir = '../nats_results/attn_vis'
        if not os.path.exists(vis_dir):
            os.mkdir(vis_dir)
        else:
            shutil.rmtree(vis_dir)
            os.mkdir(vis_dir)
        
        for model_name in self.base_models: 
            self.base_models[model_name].eval()
        for model_name in self.train_models: 
            self.train_models[model_name].eval()
        with torch.no_grad():
            
            print('Begin Visualization')
            n_batch = len(self.vis_data)
            print('The number of batches (visualization): {}'.format(n_batch))
            for batch_id in range(n_batch):

                self.build_batch(self.vis_data[batch_id])
                self.visualization_worker(batch_id, vis_dir)

                show_progress(batch_id+1, n_batch)
            print()
                        
