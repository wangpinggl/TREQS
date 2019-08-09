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
        accu = accuracy_score(true_data, pred_data)

        print('Accuracy={}'.format(np.round(accu, 4)))
        mem_score['validate']= accu
        if accu > score_validate:
            score_validate = accu
            mdx_validate = epoch

        pred_data = np.loadtxt('../nats_results/test_pred_{}.txt'.format(epoch))
        true_data = np.loadtxt('../nats_results/test_true_{}.txt'.format(epoch))

        accu = accuracy_score(true_data, pred_data)
        print('Accuracy={}'.format(np.round(accu, 4)))
        mem_score['test'] = accu

        if accu > score_test:
            score_test = accu
            mdx_test = epoch

        memo.append(mem_score)

    print('='*50)
    print('Best epoch {}'.format(mdx_validate))
    print('='*50)
    print('Val')
    accu = memo[mdx_validate-1]['validate']
    print('Accuracy={}'.format(np.round(accu, 4)))
    print('Test')
    accu = memo[mdx_validate-1]['test']
    print('Accuracy={}'.format(np.round(accu, 4)))
    print('='*50)
    print('Max epoch {}'.format(mdx_test))
    print('='*50)
    print('Val')
    accu = memo[mdx_test-1]['validate']
    print('Accuracy={}'.format(np.round(accu, 4)))
    print('Test')
    accu = memo[mdx_test-1]['test']
    print('Accuracy={}'.format(np.round(accu, 4)))
    