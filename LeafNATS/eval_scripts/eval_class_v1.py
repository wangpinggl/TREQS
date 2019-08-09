import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

def evaluation(args):
    '''
    We use f-score, accuracy, MSE to evaluation the performance of different models.
    Here, the best model is selected based on the averaged f-score.
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

        (p1, r1, f1, _) = precision_recall_fscore_support(true_data, pred_data, average='macro')
        accu = accuracy_score(true_data, pred_data)
        mse = mean_squared_error(true_data, pred_data)

        print('f_score={}, Accuracy={}, MSE={}'.format(
            np.round(f1, 4), np.round(accu, 4), np.round(mse, 4)))
        mem_score['validate']= [p1, r1, f1, accu, mse]

        pred_data = np.loadtxt('../nats_results/test_pred_{}.txt'.format(epoch))
        true_data = np.loadtxt('../nats_results/test_true_{}.txt'.format(epoch))

        if accu > score_validate:
            score_validate = accu
            mdx_validate = epoch

        (p1, r1, f1, _) = precision_recall_fscore_support(true_data, pred_data, average='macro')
        accu = accuracy_score(true_data, pred_data)
        mse = mean_squared_error(true_data, pred_data)
        print('f_score={}, Accuracy={}, MSE={}'.format(
            np.round(f1, 4), np.round(accu, 4), np.round(mse, 4)))
        mem_score['test'] = [p1, r1, f1, accu, mse]

        if accu > score_test:
            score_test = accu
            mdx_test = epoch

        memo.append(mem_score)

    print('='*50)
    print('Best epoch {}'.format(mdx_validate))
    print('='*50)
    print('Val')
    [p1, r1, f1, accu, mse] = memo[mdx_validate-1]['validate']
    print('f_score={}, Accuracy={}, MSE={}'.format(
        np.round(f1, 4), np.round(accu, 4), np.round(mse, 4)))
    print('Test')
    [p1, r1, f1, accu, mse] = memo[mdx_validate-1]['test']
    print('f_score={}, Accuracy={}, MSE={}'.format(
        np.round(f1, 4), np.round(accu, 4), np.round(mse, 4)))
    print('='*50)
    print('Max epoch {}'.format(mdx_test))
    print('='*50)
    print('Val')
    [p1, r1, f1, accu, mse] = memo[mdx_test-1]['validate']
    print('f_score={}, Accuracy={}, MSE={}'.format(
        np.round(f1, 4), np.round(accu, 4), np.round(mse, 4)))
    print('Test')
    [p1, r1, f1, accu, mse] = memo[mdx_test-1]['test']
    print('f_score={}, Accuracy={}, MSE={}'.format(
        np.round(f1, 4), np.round(accu, 4), np.round(mse, 4)))