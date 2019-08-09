'''
@author Tian Shi
Please contact tshi@vt.edu
'''
import numpy as np
from .utils import eval_dmsc

def evaluation(args):
    '''
    Evaluation Metrics
    -- Accuracy, MSE
    Best model is selected based on accuracy.
    '''
    score_test = 0.0
    score_validate = 0.0
    mdx_test = 1
    mdx_validate = 1
    memo = []
    for epoch in range(1, args.n_epoch+1):
        print('='*50)
        print('Epoch: {}'.format(epoch))
        score_dict = {}

        mem_score = {'validate': [], 'test': []}

        pred_data = np.loadtxt('../nats_results/validate_pred_{}.txt'.format(epoch))
        true_data = np.loadtxt('../nats_results/validate_true_{}.txt'.format(epoch))
        n_tasks = pred_data.shape[1]

        label_pred = []
        label_true = []
        for k in range(n_tasks):
            predlb = [rt for idx, rt in enumerate(pred_data[:, k].tolist()) if true_data[idx, k] != 0]
            truelb = [rt for idx, rt in enumerate(true_data[:, k].tolist()) if true_data[idx, k] != 0]
            label_pred += predlb
            label_true += truelb
            
        accu, mse = eval_dmsc(label_pred, label_true)
        mem_score['validate'].append([accu, mse])

        print('accuracy={}, MSE={}'.format(np.round(accu, 4), np.round(mse, 4)))
        if accu > score_validate:
            score_validate = accu
            mdx_validate = epoch
        
        pred_data = np.loadtxt('../nats_results/test_pred_{}.txt'.format(epoch))
        true_data = np.loadtxt('../nats_results/test_true_{}.txt'.format(epoch))

        label_pred = []
        label_true = []
        for k in range(n_tasks):
            predlb = [rt for idx, rt in enumerate(pred_data[:, k].tolist()) if true_data[idx, k] != 0]
            truelb = [rt for idx, rt in enumerate(true_data[:, k].tolist()) if true_data[idx, k] != 0]
            label_pred += predlb
            label_true += truelb
            
        accu, mse = eval_dmsc(label_pred, label_true)
        mem_score['test'].append([accu, mse])

        print('accuracy={}, MSE={}'.format(np.round(accu, 4), np.round(mse, 4)))
        if accu > score_test:
            score_test = accu
            mdx_test = epoch

        memo.append(mem_score)

    print('='*50)
    print('Best epoch {}'.format(mdx_validate))
    print('='*50)
    out = []
    [accu, mse] = memo[mdx_validate-1]['validate'][0]
    accu = np.round(accu, 4)
    mse = np.round(mse, 4)
    out.append(accu)
    out.append(mse)
    print('accuracy={}, MSE={}'.format(accu, mse))
    [accu, mse] = memo[mdx_validate-1]['test'][0]
    accu = np.round(accu, 4)
    mse = np.round(mse, 4)
    out.append(accu)
    out.append(mse)
    print('accuracy={}, MSE={}'.format(accu, mse))
    print(' '.join(list(map(str, out))))
    
    print('='*50)
    print('Max epoch {}'.format(mdx_test))
    print('='*50)
    out = []
    [accu, mse] = memo[mdx_test-1]['validate'][0]
    accu = np.round(accu, 4)
    mse = np.round(mse, 4)
    out.append(accu)
    out.append(mse)
    print('accuracy={}, MSE={}'.format(accu, mse))
    [accu, mse] = memo[mdx_test-1]['test'][0]
    accu = np.round(accu, 4)
    mse = np.round(mse, 4)
    out.append(accu)
    out.append(mse)
    print('accuracy={}, MSE={}'.format(accu, mse))
    print(' '.join(list(map(str, out))))