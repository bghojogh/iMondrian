# %matplotlib inline
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import sklearn.metrics as metrics
from scipy import stats
from scipy.io import loadmat
from sklearn import model_selection
from sklearn import preprocessing

from imondrian.mondrianforest import MondrianForest
from imondrian.mondrianforest import process_command_line
from imondrian.mondrianforest_utils import precompute_minimal
from imondrian.mondrianforest_utils import reset_random_seed
# from imondrian.mondrianforest_utils import load_data

warnings.filterwarnings('ignore')


class PDF:
    """
    Display PDF in jupyter
    https://stackoverflow.com/a/19470377/6923716
    """

    def __init__(self, pdf, size=(200, 200)):
        self.pdf = pdf
        self.size = size

    def _repr_html_(self):
        return '<iframe src={0} width={1[0]} height={1[1]}></iframe>'.format(self.pdf, self.size)

    def _repr_latex_(self):
        return r'\includegraphics[width=1.0\textwidth]{{{0}}}'.format(self.pdf)


def format_data_dict(X, y):
    return {
        'is_sparse': False,
        'n_class': 2,
        'n_dim': len(X[0]),
        'n_train': len(X),
        'n_test': 0,
        'train_ids_partition': {
            'cumulative': {0: np.array(list(range(len(X))))},
            'current': {0: np.array(list(range(len(X))))}
        },
        'x_train': X,
        'y_train': y,
        'x_test': np.array([]),
        'y_test': np.array([]),
    }


def load_mat(path):
    """
    load matlab files
    format: X = Multi-dimensional point data, y = labels (1 = outliers, 0 = inliers)
    """
    data = loadmat(path)
    return data['X'].astype(float), data['y'].flatten().astype(int)


def load_from_csv(file, start=1, end=None, label=0):

    data = np.loadtxt(file, dtype=np.str, delimiter=',')[1:]
    X = data[:, start:end].astype(np.float)
    y = data[:, label]

    # label encode
    le = preprocessing.LabelEncoder()
    le.fit(['"anomaly"', '"nominal"'])
    y = le.transform(y)

    return X, y


def load_from_dir(path, count=None, start=1, end=None, label=0):
    X, y = [], []
    # path = '../Algorithms_for_comparison/' + path
    file_count = 0
    for file in os.listdir(path):
        if file.endswith('.csv'):
            X_part, y_part = load_from_csv(path+'/'+file, start, end, label)
            print('Loading', file, X_part, y_part)
            # simple append data
            X.extend(X_part)
            y.extend(y_part)
            print('finished.')
            if count:
                file_count = file_count + 1
                if count == file_count:
                    break
    return X, y


def train(X, y, num_trees=100):
    argv = [
        '--n_minibatches', '1',
        '--n_mondrians', str(num_trees),
        '--budget', '-1',
        '--normalize_features', '1',
        '--optype', 'class',
        '--draw_mondrian', '0',
        '--isolate',
    ]

    settings = process_command_line(argv)

    reset_random_seed(settings)

    data = format_data_dict(X, y)

    param, cache = precompute_minimal(data, settings)

    mf = MondrianForest(settings, data)

    print(('Training on %d samples of dimension %d' %
           (data['n_train'], data['n_dim'])))

    for idx_minibatch in range(settings.n_minibatches):
        train_ids_current_minibatch = data['train_ids_partition']['current'][idx_minibatch]
        if idx_minibatch == 0:
            # Batch training for first minibatch
            mf.fit(data, train_ids_current_minibatch, settings, param, cache)
        else:
            # Online update
            mf.partial_fit(
                data, train_ids_current_minibatch,
                settings, param, cache,
            )

    print('\nFinal forest stats:')
    tree_stats = np.zeros((settings.n_mondrians, 2))
    tree_average_depth = np.zeros(settings.n_mondrians)
    for i_t, tree in enumerate(mf.forest):
        tree_stats[
            i_t, -
            2:
        ] = np.array([len(tree.leaf_nodes), len(tree.non_leaf_nodes)])
        tree_average_depth[i_t] = tree.get_average_depth(settings, data)[0]
    print(('mean(num_leaves) = {:.1f}, mean(num_non_leaves) = {:.1f}, mean(tree_average_depth) = {:.1f}'.format(
        np.mean(tree_stats[:, -2]), np.mean(tree_stats[:, -1]), np.mean(tree_average_depth))))
    print(('n_train = %d, log_2(n_train) = %.1f, mean(tree_average_depth) = %.1f +- %.1f' %
           (data['n_train'], np.log2(data['n_train']), np.mean(tree_average_depth), np.std(tree_average_depth))))

    return mf, settings, data


def evaluate(y_true, y_score, iter):
    fpr, tpr, threshold = metrics.roc_curve(y_true, np.asarray(y_score))
    roc_auc = metrics.auc(fpr, tpr)
    if iter == 0:
        print('AUC = ', roc_auc)
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.4f' % roc_auc)
        # plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
    return roc_auc


def load_from_mat(mat_fname):
    mat_contents = sio.loadmat(mat_fname)

    X = mat_contents['X']
    y = mat_contents['y'].astype(int).ravel()
    y = 1 - y
    return X, y


datasets = {'cardio.mat', 'shuttle.mat', 'thyroid.mat'}
datasets = {'mnist.mat'}
iterations = 5
ratio = 0.7
for mat_fname in datasets:
    # DATASET_FILE = '/Users/haoranma/Desktop/research/'
    DATASET_FILE = ''
    DATASET_FILE += mat_fname

    print('================================================================')
    print('> dataset: ', mat_fname)
    print('================================================================')
    X, y = load_mat(DATASET_FILE)

    if(mat_fname == 'mnist.mat'):
        # has many attribute with no information
        # remove these attributes here
        # has 76/100 attributes remaining
        X = X[:, [1, 2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23, 24, 25, 27,
                  29, 30, 31, 32, 33, 34, 35, 36, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51,
                  54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68, 69, 71, 73, 74, 75, 76, 77,
                  79, 80, 81, 82, 83, 84, 85, 90, 92, 93, 94, 95, 96, 97, 98]]
    X = stats.zscore(X)
    # zscore normalize for imondrian to perform better
    # also allows SVM to converge and terminate

    '''
    # for testing with older datasets
    # need to replace the new one to work
    # cannot simply uncomment
    datasets = {'skinseg','imgseg','abalone','magicgamma'}
    datasets = {'pageb','synthetic','shuttle','particle','spambase','kddcup99'}
    for csv_dir in datasets:
        if csv_dir == 'skinseg':
            X, y = load_from_dir(csv_dir,1,0,2,3)
        elif csv_dir == 'imgseg':
            X, y = load_from_dir(csv_dir,1,0,17,18)
        elif csv_dir == 'abalone':
            X, y = load_from_dir(csv_dir,1,0,6,7)
        elif csv_dir == 'magicgamma':
            X, y = load_from_dir(csv_dir,1,0,9,10)
        elif csv_dir == 'pageb':
            X, y = load_from_dir(csv_dir,1,0,9,10)
        elif csv_dir == 'synthetic':
            X, y = load_from_dir(csv_dir,1,0,9,10)
        elif csv_dir == 'shuttle':
            X, y = load_from_dir(csv_dir,1,0,8,9)
        elif csv_dir == 'particle':
            X, y = load_from_dir(csv_dir,1,0,49,50)
        elif csv_dir == 'spambase':
            X, y = load_from_dir(csv_dir,1,0,56,57)
        elif csv_dir == 'kddcup99':
            X, y = load_from_dir(csv_dir,1,0,2,3)
        print('> dataset: ', csv_dir)
        print('================================================================')
    '''

    X = np.asarray(X)
    y = np.asarray(y)

    auc = []
    for i in range(iterations):
        X_train, X_test, y_train, y_test = \
            model_selection.train_test_split(
                X, y, test_size=(1-ratio), shuffle=True)
        mf, settings, data = train(X_train, y_train, num_trees=100)

        auc.append(evaluate(y_test, mf.get_anomaly_scores(X_test), i))
    auc = np.asarray(auc)
    print('AUC = ', np.mean(auc))
    print('================================================================')
