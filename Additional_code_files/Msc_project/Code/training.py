import pickle
import xgboost as xgb
from sklearn.model_selection import *
from sklearn.metrics import *
from pandas import *
import pandas as pd
from sklearn.utils import resample
import multiprocessing
import warnings

cpu_count = multiprocessing.cpu_count()
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_pickle_file(pickled_file):
    print(f'Loading data file from {pickled_file}')
    infile = open(pickled_file, 'rb')
    unpickled_file = pickle.load(infile)
    print(f'Loaded {len(unpickled_file)} entries')
    infile.close()
    return unpickled_file


def save_pickle_file(path, data):
    print('Dumping data to path {}'.format(path))
    with open(path, 'wb') as file:
        pickle.dump(data, file)
    print('Finished dumping data to path {}'.format(path))


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="red" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


def prepare_data(initial_features):
    df = initial_features

    df['label'] = df['infected_status'].apply(lambda x: 1 if x == True else 0)
    df = df.reset_index(drop=True)

    df = df.drop(columns=['user_id', 'infected_status', 'infection_time', 'followers_list'], axis=1)

    # Converting all type to float, to prepare for feature selection
    df = df.astype('float')
    # Reset index, with drop equals to true to avoid setting old index as a new column
    df = df.reset_index(drop=True)
    # Visualize distribution
    #print('[Original] data counts, with uninfected (0): {}, infected (1): {}'.format(
        # df['label'].value_counts()[0],
        # df['label'].value_counts()[1]
    # )
    # )
    df.groupby(['TwM_tCurrent', 'label']).size().unstack(fill_value=0).plot.bar(title='Original Data Distribution')

    columns = list(df.columns)
    columns.remove('label')

    X = df[columns]
    y = df[['label']]
    return df, X, y


def upsample(df):
    # Separate majority and minority classes
    df_majority = df[df.label == 0]  # Uninfected is the major class
    df_minority = df[df.label == 1]  # Infected is the minor class

    # Upsample minority class
    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=len(df_majority),  # to match majority class
                                     random_state=123)  # reproducible results

    # Combine majority class with upsampled minority class
    df_upsampled = pd.concat([df_majority, df_minority_upsampled])

    # Display new class counts
    #print(df_upsampled.label.value_counts())

    return df_upsampled


def train(df, X, y, params, n_folds, num_boost_round, rebalance_method):
    # 2. N Fold Split
    # Stratified K-Folds cross-validator
    # Provides train/test indices to split data in train/test sets.
    # This cross-validation object is a variation of KFold that returns stratified folds.
    # The folds are made by preserving the percentage of samples for each class.
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

    corrDataframe = pd.DataFrame()
    mse = []
    acc = []
    roc = []
    F1 = []
    auc = []
    auc_t = []
    acc_t = []
    fold_count = 0
    t_current = 210
    number_of_features = len(X.columns)

    #print("Start cross validation")
    for train, test in skf.split(X, y):
        #print("===Processing fold %s===" % fold_count)
        train_fold = df.loc[train]
        test_fold = df.loc[test]

        # 3. Rebalance
        if rebalance_method == 'up':
            train_fold = upsample(train_fold)
        if rebalance_method == 'down':
            train_fold = downsample(train_fold)

        # 4. Feature Selection
        #         corr = train_fold.corr()['label'][train_fold.corr()['label'] < 1].abs()
        #         corr = corr.sort_values(ascending=False)
        #         corrDataframe = corrDataframe.append(pd.DataFrame(corr.rename('cv{}'.format(fold_count))).T)
        #         features = corr.index[range(number_of_features)].values
        features = X.columns

        # 5. Training
        # Fit Model
        xgtrain = xgb.DMatrix(train_fold[features].values, train_fold['label'].values)
        xgtest = xgb.DMatrix(test_fold[features].values, test_fold['label'].values)
        evallist = [(xgtrain, 'train'), (xgtest, 'eval')]
        #         evallist = []

        bst = xgb.train(params, xgtrain,
                        num_boost_round=num_boost_round,
                        evals=evallist)

        # 6. Testing
        # Check MSE on test set

        test_fold_t = test_fold[test_fold.TwM_tCurrent == t_current]
        #             xgtest = xgb.DMatrix(test_fold[features].values)
        xgtest_t = xgb.DMatrix(test_fold_t[features].values)
        pred = bst.predict(xgtest)
        pred_t = bst.predict(xgtest_t)

        mse.append(mean_squared_error(test_fold['label'], pred))
        roc.append(roc_auc_score(test_fold['label'], pred))
        # auc_t.append(roc_auc_score(test_fold_t['label'], pred_t))

        acc.append(accuracy_score(test_fold['label'], (pred > 0.5).astype(int)))
        acc_t.append(accuracy_score(test_fold_t['label'], (pred_t > 0.5).astype(int)))
        F1.append(f1_score(test_fold['label'], (pred > 0.5).astype(int)))
        cm = confusion_matrix(test_fold['label'], (pred > 0.5).astype(int))
        plot_confusion_matrix(cm,
                              normalize=True,
                              target_names=['Uninfected', 'Infected'],
                              title="Confusion Matrix, Normalized")

        fold_count += 1
        # Done with the fold
    # print("Finished cross validation")
    # print("MSE: {} ".format(DataFrame(mse).mean()))
    # print("ACC: {} ".format(DataFrame(acc).mean()))
    # print("AUC: {} ".format(DataFrame(roc).mean()))
    # #     print("F1: {} ".format(DataFrame(F1).mean()))
    # print("ACC for t at {}: {} ".format(t_current, DataFrame(acc_t).mean()))
    corrDataframe = corrDataframe.T
    corrDataframe['average corr'] = corrDataframe.mean(numeric_only=True, axis=1)
    #print(corrDataframe.sort_values(by=['average corr'], ascending=False).to_string())

    return bst


def training_procedure(start_time):
    path = "/Users/jay/MSC_WSBDA/MSc_Thesis/Msc_project/Data/"
    event = 'givenchy'
    start_hour = start_time

    initial_features = load_pickle_file(path+str(start_hour)+'_hrs_data.pkl')
    users = load_pickle_file(path+event+"_users.dat")
    users.reset_index(drop =True , inplace =True)
    df, X, y = prepare_data(initial_features)
    param = {'max_depth':3,'eta': 0.1,'gamma':10,'min_child_weight':10,'silent': 1,'objective': 'binary:logistic','subsample': 0.9}
    param['nthread'] = cpu_count
    param['eval_metric'] = ['auc']
    num_boost_round = 1000
    rebalance_method = 'up'

    columns = list(df.columns)
    columns.remove('label')
    if rebalance_method == 'up':
        df_rebalance = upsample(df)

    X = df_rebalance[columns]
    y = df_rebalance[['label']]

    # load JS visualization code to notebook


    # train XGBoost model

    model = xgb.train(param, xgb.DMatrix(X, label=y), num_boost_round,verbose_eval=False)

    # explain the model's predictions using SHAP values
    # (same syntax works for LightGBM, CatBoost, and scikit-learn models)

    return model