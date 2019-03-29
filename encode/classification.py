import sys, os
import argparse
import numpy as np
import copy
from ast import literal_eval
import progressbar

# different classifiers
from sklearn.svm import LinearSVC, SVC, OneClassSVM, SVR, LinearSVR
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,\
                            GradientBoostingClassifier, AdaBoostClassifier,\
                        RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import ParameterGrid, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV
import evaluate
import puhma_common as pc
import preprocess
from sklearn import preprocessing, cross_validation
from sklearn.metrics import confusion_matrix, classification_report,\
        recall_score, roc_auc_score, f1_score,\
        precision_recall_fscore_support,\
        roc_curve, accuracy_score, average_precision_score,\
        make_scorer, r2_score, mean_squared_error, get_scorer,\
        mean_absolute_error, cohen_kappa_score
import exemplar_cls, exemplar_cls_quwi
from scipy import stats
try:
    from xgboost.sklearn import XGBClassifier
except:
    print 'xgboost classifier not available'


def class_weight(s):
    if 'None' in s:
        return None
    if 'auto' in s:
        return 'auto'
    if 'balanced' in s:
        return 'balanced'
    s = s.strip(' \t\n\r') # probably argparse does that already
    pairs = s.split(',')
    d = {}
    for p in pairs:
        if p == '': continue
        c,v = p.split(':')
        d[int(c)] = float(v)
    if len(d.keys()) == 0:
        raise argparse.ArgumentTypeError('wrong class weight arguments, need'
                                         ' format c:v')
    return d

def getSVM(args):
    gamma = args.gamma if args.gamma == 'auto' else float(args.gamma)

    if args.regression:
        cls = SVR(C=args.C, kernel=args.kernel,
                  epsilon=args.epsilon,
                  gamma=gamma)
    else:
        if args.ovo:
            cls = SVC(C=args.C, kernel=args.kernel,
                      gamma=gamma,
                      probability=args.proba,
                      class_weight=args.class_weight,
                     decision_function_shape='ovr' # let the shape be ovr
                     )
        else:
            if args.parallel: n_jobs = args.nprocs if args.nprocs else -1
            else: n_jobs = 1
            cls = OneVsRestClassifier(SVC(C=args.C, kernel=args.kernel, gamma=gamma,
                                         probability=args.proba,
                                         class_weight=args.class_weight),
                                     n_jobs=n_jobs)

    if args.grid:
        C_range = 10.0**np.arange(args.C_range[0], args.C_range[1])

        if args.regression:
            epsilon_range = 10.0**np.arange(args.epsilon_range[0], args.epsilon_range[1])

            grid = [{'C': C_range, 'epsilon': epsilon_range}]
        else:
            gamma_range = [gamma]

            if args.gamma_range:
                gamma_range.extend(10.0**np.arange(args.gamma_range[0], args.gamma_range[1]))

            if isinstance(cls, OneVsRestClassifier):
                grid = { 'estimator__C': C_range }

                if gamma_range is not None:
                    grid['estimator__gamma'] = gamma_range
            else:
                grid = { 'C': C_range }

                if gamma_range is not None:
                    grid['gamma'] = gamma_range

        return cls, grid
    return cls, None

def getRandomForest(args):
    if args.parallel:
        n_jobs = args.nprocs if args.nprocs else -1
    else:
        n_jobs = 1

    if args.regression:
        cls = RandomForestRegressor(n_estimators=args.trees,
                                  max_depth=args.depth,
                                  n_jobs=n_jobs)
    else:
        cls = RandomForestClassifier(n_estimators=args.trees,
                                  max_depth=args.depth,
                                  n_jobs=n_jobs)

    if args.grid:
        grid = [{'max_depth': [3,10,25],
                 'n_estimators': [5,25,50,100]} ]
        return cls, grid
    return cls, None

def getGaussianNB(args):
    return [ GaussianNB() ]

def getKNN(args):
    if args.parallel:
        n_jobs = args.nprocs if args.nprocs else -1
    else:
        n_jobs = 1
    cls = KNeighborsClassifier(n_neighbors=args.k,
                               weights='uniform' \
                               if args.distance == False \
                               else 'distance',
                               n_jobs=n_jobs)
    if args.grid:
        grid = [{'n_neighbors':[3,5,7],
                 'weights':['uniform', 'distance']} ]
        return cls, grid
    return cls, None

def getLDA(args):
    return [ LDA() ]

def getQDA(args):
    return [ QDA() ]

def getAdaBoost(args):
    cls = AdaBoostClassifier(n_estimators=args.n_estimators)
    if args.grid:
        grid = [{'n_estimators': [50,100,250]} ]
        return cls, grid
    return cls, None

def getLSVM(args):
    if args.regression:
        cls = LinearSVR(dual=args.dual, C=args.C,
                        epsilon=args.epsilon,
                        fit_intercept=args.fit_intercept)
    else:
        cls = LinearSVC(dual=args.dual, C=args.C,
                       class_weight=args.class_weight,
                       fit_intercept=args.fit_intercept)

    if args.grid:
        C_range = 10.0**np.arange(args.C_range[0], args.C_range[1])

        if args.regression:
            epsilon_range = 10.0**np.arange(args.epsilon_range[0], args.epsilon_range[1])

            grid = [{'C': C_range, 'epsilon': epsilon_range}]
        else:
            grid = { 'C': C_range }

        """
        grid = [{'loss': ['l2'],
                 'penalty':['l2'],
                 'dual': [True,False],
                'C': np.logspace(-4,3,8)},
                # penalty='l2' and loss='l1' is only
                # supported when dual='true'
                {'loss': ['l1'],
                 'penalty':['l2'],
                 'dual': [True],
                'C': np.logspace(-4,3,8)},
                # penalty='l1' is only supported
                # when dual='false'
                # The combination of penalty='l1' and
                # loss='l1' is not supported
                {'loss': ['l2'],
                 'penalty':['l1'],
                 'dual': [False],
                'C': np.logspace(-4,3,8)}]
        """
        return cls, grid

    return cls, None

def getSGD(args):
    if args.parallel:
        n_jobs = args.nprocs if args.nprocs else -1
    else:
        n_jobs = 1

    # shuffle=True seems to improve the results -> double check that
    cls = SGDClassifier(alpha=args.alpha,
                        n_iter=args.n_iter, shuffle=False, n_jobs=n_jobs,
                        loss=args.loss, class_weight=args.class_weight)
    if args.grid:
        grid = [ #{'alpha': 10.0**-np.arange(1,7)}]
                {'alpha': 10.0**-np.arange(-1,7)
#,
#                'shuffle':[True,False],
#                'loss':['hinge','log','modified_huber'],
#                'penalty':['l2','l1','elasticnet']
                     }]
        return cls, grid

    return cls, None

def getLR(args):
    if args.parallel:
        n_jobs = args.nprocs if args.nprocs else -1
    else:
        n_jobs = 1
    if args.multi_class == 'multinomial' and args.solver not in ['newton-cg', 'lbfgs']:
        raise ValueError('multinomial only with newton-cg or lbfgs possible')
    if args.penalty == 'l1' and args.solver is not 'liblinear':
        raise ValueError('l1 only w. liblinear as solver')

    cls = LogisticRegression(C=args.C, dual=args.dual, solver=args.solver, n_jobs=n_jobs,
                               multi_class=args.multi_class,
                               penalty=args.penalty, class_weight=args.class_weight)
    if args.grid:
        grid = [ {'C': 10.0**-np.arange(-4,4)
#                'loss':['l1','l2'],
                     }]
        return cls, grid
    return cls, None

def getOneClassSVM(args):
    cls = OneClassSVM(kernel='linear', nu=args.nu)
    if args.grid:
        grid = [{'nu': np.arange(10,3,-1) / 10.0}]
        return cls, grid
    return cls, None

def getXGBoost(args):
    cls = XGBClassifier()
    if args.grid:
        grid = [{'max_depth': [2,4,6],
                 'n_estimators': [50,100,200]}]
        return cls, grid
    return cls, None

def getDummy(args):
    return None, None

def parseArguments(parser):
    parser.add_argument('--load_cls',
                        help='load cls')
    parser.add_argument('--distances', action='store_true',
                        help='compute distances')
    parser.add_argument('--taskname',
                        help='taskname for distance matrix')
    parser.add_argument('--labelfile2',
                        help='labelfile for multilabel classific')
    parser.add_argument('--dump_svmlight',action='store_true',
                        help='just dump svmlight feature/label files in the current'
                        'folder - no classification')
    parser.add_argument('--save_cls', action='store_true',
                        help='store the best (or only) classifier')
    parser.add_argument('--class_weight',
                        default='balanced',
                        type=class_weight,
                        help='class weight(s) as list, default=balanced')
    parser.add_argument('--regression', action='store_true',
                        help='regression variant of the classifier (if'
                        ' possible)')
    parser.add_argument('--absvalue', action='store_true',
                        help='turn float regression values to absvalues')
    parser.add_argument('--scorer',
                        help='use specific scoring function')
    parser.add_argument('--min_pred',
                        help='min possible predicted value for regr. auto'
                        'denotes reading from training data')
    parser.add_argument('--max_pred',
                        help='max possible predicted value for regr. auto'
                        'denotes reading from training data')
    parser.add_argument('--my_grid_cv', action='store_true',
                        help='my own grid-cv implementation with non-avg'
                        ' scoring')

    parser.add_argument('--vl',
                        help='validation labels, replaces inner-cv')
    parser.add_argument('--vi',
                        help='validation inputfolder, replaces innter-cv')
    parser.add_argument('--tl',
                        help='test labels, replaces outer-cv')
    parser.add_argument('--ti',
                        help='test inputfolder, replaces outer-cv')
    parser.add_argument('--use_ben_as_test', action='store_true',
                        help='used in run_pipeline if the training =exp'
                        'folder is only used for the vocab training '
                        '--> inner and outer cv')
    parser.add_argument('--use_exp_as_test', action='store_true',
                        help='do it on your own...')

    cv_group = parser.add_argument_group('Cross validation options')
    cv_group.add_argument('--outer_cv', choices=['lolo','lklo','strat', 'kfold',
                                                'loo'],
                          default='kfold',
                          help='outer cross-validation (no test labels (--tl) needed)')
    cv_group.add_argument('--outer_nfolds', type=int, default=5,
                          help='Number of outer CV folds')
    cv_group.add_argument('--cv', choices=['lolo', 'lklo', 'strat', 'kfold', 'loo'],
                          default='strat',
                         help='inner cross-validation')
    cv_group.add_argument('--nfolds', type=int, default=5,
                          help="Number of inner CV folds")
    cv_group.add_argument('--grid', default=False,
                          action='store_true',
                          help='use a grid-search to determine best parameters'
                          '(if available)')
    cv_group.add_argument('--proba', action='store_true', default=False,
                            help='scores are normalized via Platt')
    cv_group.add_argument('--refit', default='mode',
                          choices=['mode', 'score'],
                          help='refit if you have an innercv,'
                          ' then evaluate again after refit - kinda cheating')

    subparsers = parser.add_subparsers(dest='clsname')

    parser_svm = subparsers.add_parser('svm')
    parser_svm.add_argument('--C', default=1.0, type=float, help="C")
    parser_svm.add_argument('--kernel', default='linear',
                           help='Kernel type of the SVM')
    parser_svm.add_argument('--gamma', default='auto', help='gamma')
    parser_svm.add_argument('--ovo', action='store_true',
                            help='use one vs one classifier')
    parser_svm.add_argument('--epsilon', default=0.1, type=float,
                            help='epsilon')
    parser_svm.add_argument('--C-range', nargs=2, default=[-5.0, 5.0], type=float,
                            metavar=('LOWER', 'UPPER'),
                            help='range of the C parameter used for grid search')
    parser_svm.add_argument('--gamma-range', nargs=2, default=[], type=float,
                            metavar=('LOWER', 'UPPER'),
                            help='range of the gamma parameter used for grid search')
    parser_svm.add_argument('--epsilon-range', nargs=2, default=[-4.0, 2.0], type=float,
                            metavar=('LOWER', 'UPPER'),
                            help='range of the epsilon parameter used for grid search')
    parser_svm.set_defaults(func=getSVM)

    parser_rf = subparsers.add_parser('rf')
    parser_rf.add_argument('--trees', default=25, type=int,
                           help='Number of trees in the Random Forest')
    parser_rf.add_argument('--depth', default=None, type=int,
                           help='Maximum depth of the trees')
    parser_rf.add_argument('--parallel', action='store_true',
                           help='Train and evaluate forest in parallel')
    parser_rf.set_defaults(func=getRandomForest)

    parser_gnb = subparsers.add_parser('gnb')
    parser_gnb.set_defaults(func=getGaussianNB)

    parser_knn = subparsers.add_parser('knn')
    parser_knn.add_argument('--k', type=int, default=3,
                            help='how many nearest neighbors to take into'
                            ' account')
    parser_knn.add_argument('--distance', default=False, action='store_true',
                            help='weight by distances')
    parser_knn.set_defaults(func=getKNN)

    parser_lda = subparsers.add_parser('lda')
    parser_lda.set_defaults(func=getLDA)

    parser_qda = subparsers.add_parser('qda')
    parser_qda.set_defaults(func=getQDA)

    parser_adaboost = subparsers.add_parser('adaboost')
    parser_adaboost.add_argument('--n_estimators', type=int, default=100,
                                 help='num weak learners')
    parser_adaboost.set_defaults(func=getAdaBoost)

    parser_lsvm = subparsers.add_parser('lsvm')
    parser_lsvm.add_argument('--C', default=1.0, type=float,
                             help='C parameter for lib svm')
    parser_lsvm.add_argument('--epsilon', default=0.1, type=float,
                            help='epsilon')
    parser_lsvm.add_argument('--dual', default=True, type=literal_eval,
                             help='Prefer dual=False when n_samples >'
                             ' n_features')
    parser_lsvm.add_argument('--C-range', nargs=2, default=[-5.0, 5.0], type=float,
                            metavar=('LOWER', 'UPPER'),
                            help='range of the C parameter used for grid search')
    parser_lsvm.add_argument('--epsilon-range', nargs=2, default=[-4.0, 2.0], type=float,
                            metavar=('LOWER', 'UPPER'),
                            help='range of the epsilon parameter used for grid search')
    parser_lsvm.add_argument('--fit_intercept', type=literal_eval,
                             default=True,
                             help='fit intercept (for non-scaled data)[def:true]')
    parser_lsvm.set_defaults(func=getLSVM)

    parser_sgd = subparsers.add_parser('sgd')
    parser_sgd.add_argument('--alpha', type=float, default=0.0001,
                            help='regularization weight')
    parser_sgd.add_argument('--n_iter', type=int, default=5,
                            help='number of iterations over the training data'
                            ' (=epochs)')
    parser_sgd.add_argument('--subfolds', type=int, default=1,
                            help='split training set of current fold further'
                            'and partially fit each subfold')
    parser_sgd.add_argument('--loss', default='hinge',
                            choices=['hinge', 'log', 'modified_huber',
                                     'squared_hinge',
                                     'perceptron', 'squared_loss', 'huber',
                                     'epsilon_insensitive',
                                     'squared_epsilon_insensitive'],
                            help='choose different loss function')
    parser_sgd.set_defaults(func=getSGD)

    # logistic regression
    parser_lr = subparsers.add_parser('logr')
    parser_lr.add_argument('--C', default=1.0, type=float,
                           help='C parameter')
    parser_lr.add_argument('--penalty', default='l2',
                           choices=['l2', 'l1'],
                           help='choose the loss function')
    parser_lr.add_argument('--solver', default='sag',
                           choices=['newton-cg', 'lbfgs', 'liblinear', 'sag'],
                           help='which solver to use')
    parser_lr.add_argument('--multi_class', default='ovr',
                           choices=['ovr', 'multinomial'],
                           help='multinomial works only for lbfgs or newton-cg')
    parser_lr.add_argument('--dual', action='store_true',
                             help='Prefer dual=False when n_samples >'
                             ' n_features')
    parser_lr.set_defaults(func=getLR)

    parser_osvm = subparsers.add_parser('osvm')
    parser_osvm.add_argument('--nu', default=0.5, type=float,
                             help='nu parameter')
    parser_osvm.set_defaults(func=getOneClassSVM)

    parser_xgboost = subparsers.add_parser('xgboost')
    parser_xgboost.set_defaults(func=getXGBoost)

    parser_dummy = subparsers.add_parser('dummy')
    parser_dummy.set_defaults(func=getDummy)

    return parser

def MAP(y_true, y_scores):
    lb = preprocessing.LabelBinarizer()
    y_true = lb.fit_transform(y_true)
    #y_p = lb.transform(y_scores)

#    results = []
#    for i in range(y.shape[1]):
#        results.append( average_precision_score(y[:,i], y_p[:,i]) )

    pairs = zip(y_true, y_scores)
    results = [ average_precision_score(y, y_score) for (y, y_score)
               in pairs ]
    return np.mean(results)

def getCV(labels, cv_type, nfolds):
    if cv_type == 'lolo':
        cv_folds = cross_validation.LeaveOneLabelOut(labels)
    if cv_type == 'loo':
        cv_folds = cross_validation.LeaveOneOut(len(labels))
    elif cv_type == 'lklo':
        cv_folds = cross_validation.LabelKFold(labels, n_folds=nfolds)
    elif cv_type == 'kfold':
        cv_folds = cross_validation.KFold(len(labels), n_folds=nfolds,
                                          shuffle=True)
    else: # args.outer_cv == 'strat'
        if isinstance(labels, list) or labels.ndim == 1:
            cv_folds = cross_validation.StratifiedKFold(labels, n_folds=nfolds)
        else:
            cv_folds = cross_validation.StratifiedKFold(labels[:,0], n_folds=nfolds)

    return cv_folds

def checkArray(arr, what=''):
    if np.isnan(arr).any():
        raise ValueError('nan in array! {}'.format(what))
    if np.isinf(arr).any():
        raise ValueError('inf in array! {}'.format(what))

# use spearman-correlation as scoring function
def scorer_spear(yy, y):
    sp, _ =  stats.spearmanr(yy, y)
    return sp

def myScorer(scorer, n_classes, default_regression=None):
    if scorer is None:
        if default_regression is None:
            return None
        if default_regression == True:
            print 'no scorer given, use r2'
            return get_scorer('r2')

        print 'no scorer given, use accuracy '
        return get_scorer('accuracy')

    if 'spear' in scorer:
        sc = make_scorer(scorer_spear)
        return sc

    return get_scorer(scorer)

def myGridCV(descr, labels,
             train_sample_weights,
             test_sample_weights,
             estimator, grid, cv,
             average=False, parallel=False, nprocs=None,
             scoring=None):
    """
    own cross-validation impl.
    does no refit with all labels at the end
    """

    all_combos = list(ParameterGrid(grid))
    print 'test {} combos:'.format(len(all_combos))

    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
                   progressbar.ETA()]
    progress = progressbar.ProgressBar(widgets=widgets, maxval=len(all_combos))

    if train_sample_weights is not None and test_sample_weights is not None:
        sample_weights = train_sample_weights #np.concatenate((train_sample_weights, test_sample_weights))
    else:
        sample_weights = None

    #print 'scoring', scoring

    def testCombo(i):
        test_labels = []
        all_pred = []
        all_train_sample_weights = []
        all_test_sample_weights = []

        for e, (train, test) in enumerate(cv):
            if sample_weights is not None:
                current_train_sample_weights = sample_weights[train]
                current_test_sample_weights = sample_weights[test]
            else:
                current_train_sample_weights = None
                current_test_sample_weights = None

            # maybe we can spare this deepcopy here
            cls = copy.deepcopy(estimator)
            cls.set_params(**all_combos[i])
            cls.fit(descr[train], labels[train],
                    sample_weight=current_train_sample_weights)
            pred = cls.predict(descr[test])
            test_labels.append(labels[test])

            all_pred.append(pred)
            all_train_sample_weights.append(current_train_sample_weights)
            all_test_sample_weights.append(current_test_sample_weights)

        progress.update(i+1)

        if average:
            return np.mean([scoring._score_func(yy, y, sample_weight=weights) for yy, y, weights in zip(all_pred, test_labels, all_test_sample_weights)])
        else:
            return scoring._score_func(np.concatenate(all_pred),
                    np.concatenate(test_labels),
                    sample_weight=np.concatenate(all_test_sample_weights))

    progress.start()

    if parallel:
        scores = pc.parmap(testCombo, range(len(all_combos)), nprocs)
    else:
        scores = map(testCombo, range(len(all_combos)))

    progress.finish()

    best_combo = all_combos[np.argmax(scores)]
    best_score = np.max(scores)

    cls = copy.deepcopy(estimator)

    cls.best_score_ = best_score
    return cls.set_params(**best_combo)

def thresholdPredictions(pred, labels, mini, maxi):
    if mini is not None:
        if mini == 'auto':
            m = np.min(labels)
            pred[pred < m] = m
        else:
            pred[pred < float(mini)] = float(mini)
    if maxi is not None:
        if maxi == 'auto':
            m = np.max(labels)
            pred[pred > m] = m
        else:
            pred[pred > float(maxi)] = float(maxi)

    return pred

def cls_stats(all_test, all_pred, all_dec, test_sample_weights):
    acc = accuracy_score(all_test, all_pred, sample_weight=test_sample_weights)
    print 'accuracy score:', acc

    all_dec = np.concatenate(all_dec, axis=0)
    cm = confusion_matrix(all_test, all_pred)
    print cm

    print 'decision-shape:', all_dec.shape
    if all_dec.ndim == 2:
        mAP = MAP(all_test, all_dec)
        ind = all_dec.argsort()[:,::-1]

#        dec_norm = np.abs(preprocessing.normalize(all_dec, norm='l1',
#                                                  copy=True))

        soft_list = np.zeros( (len(all_test),10), dtype=np.float32)
        for r in range(len(all_test)):
            for k in xrange(all_dec.shape[1]):
                if ind[r,k] == all_test[r]:
                    if k < 10:
                        soft_list[r,k:] = 1
                    break
        soft_list = np.mean(soft_list,axis=0)
        print 'topx-soft:',soft_list
        print 'map:', mAP

    pmic, rmic, fmic, _ = precision_recall_fscore_support(all_test,
                                             all_pred,
                                             average='micro',
                                             sample_weight=test_sample_weights)
    print 'micro:', pmic, rmic, fmic
    pmac, rmac, fmac, _ = precision_recall_fscore_support(all_test,
                                             all_pred,
                                             average='macro',
                                             sample_weight=test_sample_weights)
    print 'macro:', pmac, rmac, fmac
    kappa = cohen_kappa_score(all_test, all_pred,
                              sample_weight=test_sample_weights)
    print 'kappa:', kappa

    if all_dec.ndim == 2:
        stats_ret = {'acc':acc, 'mAP':mAP, 'topx_soft':soft_list,
                 'kappa':kappa, 'rmac':rmac, 'rmic':rmic, 'pmic':pmic,
                 'pmac':pmac, 'fmic':fmic, 'fmac':fmac}
    else:
        stats_ret = {'acc':acc, 'kappa':kappa, 'rmac':rmac,
                     'rmic':rmic, 'pmic':pmic,
                     'pmac':pmac, 'fmic':fmic, 'fmac':fmac}
    return stats_ret

def writeBelonging(pred_labels, dec, outputdir, test_files, taskname, pattern,
                   le):
    """
    needed for clamm competition 2017
    """
    n_classes = dec.shape[1]

    if taskname:
        if taskname.endswith('1') or taskname.endswith('2'):
            ty = 'SCRIPT_TYPE'
        else:
            ty = 'DATE_TYPE'
    else:
        taskname = 'classification'
        ty = 'CLASS'
    # open files
    bc = open(os.path.join(outputdir,taskname+'_belonging_class.csv'), 'w')
    bm = open(os.path.join(outputdir,taskname+'_belonging_matrix.csv'), 'w')

    # first lines
    bc.write('FILENAME,{}\n'.format(ty))
    first_line = 'FILENAME'
    for i in range(n_classes):
        first_line += ',{}{}'.format(ty,le.inverse_transform(i))
    bm.write(first_line + '\n')

    for e, fn in enumerate(test_files):
        # prediction
        bc.write('{},{}\n'.format(os.path.basename(fn).replace(pattern,''), pred_labels[e]))
        #    if all_encs[e] is None:
        #        continue
        bm.write('{}'.format(os.path.basename(fn).replace(pattern,'')))
        for i in range(n_classes):
            bm.write(',{}'.format(dec[e,i]))
        bm.write('\n')

    bc.close()
    bm.close()


def writeDistanceMatrix(test_descr, outputfolder, test_files,
                        taskname, pattern):
        print '> compute distances'
        distances = evaluate.computeDistances(test_descr, 'cosine')
        np.fill_diagonal(distances,0)
        preprocessing.normalize(distances, norm='l1', copy=False)
#        distances /= np.sum(distances)
        print '< compute distances'

        assert(distances.shape[0] == len(test_files))
        assert(distances.shape[1] == len(test_files))
        print '> write distance_matrix'
        with open(os.path.join(outputfolder,taskname+'_distance_matrix.csv'), 'w') as f:
            f.write('{},'.format(taskname))
            for k in range(distances.shape[1]):
                f.write(os.path.basename(test_files[k]).replace(pattern,''))
                f.write(',')
            f.write('\n')
            for i in range(distances.shape[0]):
                f.write(os.path.basename(test_files[i]).replace(pattern,''))
                for k in range(distances.shape[1]):
                    f.write(',{:.6f}'.format(distances[i,k]))
                f.write('\n')
        print '< write distance_matrix'

def run(args, prep=None):
    if prep is None:
        prep = preprocess.Preprocess()

    files, labels = pc.getFiles(args.inputfolder, args.suffix,
                                labelfile=args.labelfile)

    if args.train_sample_weights:
        train_sample_weights = np.genfromtxt(args.train_sample_weights)
        print 'loaded train sample weights:', train_sample_weights.shape, train_sample_weights
    else:
        train_sample_weights = None

    if args.test_sample_weights:
        test_sample_weights = np.genfromtxt(args.test_sample_weights)
        print 'loaded test sample weights:', test_sample_weights.shape, test_sample_weights
    else:
        test_sample_weights = None

    # deal with strings as labels
    if args.regression:
        print 'regression -> turn labels into floats'
        le = None
        labels = np.array(labels).astype(np.float)
    else:
        if args.labelfile2:
            files2, labels2 = pc.getFiles(args.inputfolder, args.suffix,
                                    labelfile=args.labelfile2)
            assert(len(files2) == len(files))
            assert(files == files2)
            le = preprocessing.LabelEncoder()
            labels = le.fit_transform(labels)
            le2 = preprocessing.LabelEncoder()
            labels2 = le2.fit_transform(labels2)
            labels = np.concatenate([np.array(labels).reshape(-1,1),
                                     np.array(labels2).reshape(-1,1)],axis=1)
            print 'mlb -> labels.shape:', labels.shape
        else:
            print 'classification -> encode labels'
            le = preprocessing.LabelEncoder()
            labels = le.fit_transform(labels)
            print 'classes:', le.classes_

    if args.grid:
        # perform outer and inner cv
        assert(args.vl or args.cv)
    cls, grid = args.func(args)
    print 'cls:params:', cls.get_params().keys()


    # let's assume they all fit in RAM
    descr = pc.loadDescriptors(files)
    checkArray(descr, 'full descriptors')

    print descr.shape

    # needed for parkinsons:
#    sel = np.array([2, 11, 14, 19, 21, 25, 26, 27, 30, 36, 38, 39, 43, 44, 52, 53, 54, 58, 59,
#     61, 62])
#    descr = descr[:,sel]
#    print descr.shape
#    import regressors
#    spTest, _ , C, ev, _, _ = regressors.SVR(descr, labels, 'loo')
#    print spTest, C, ev
#    sys.exit(1)

    if args.outer_cv == 'lklo' or args.outer_cv == 'lolo'\
        or args.cv == 'lklo' or args.cv == 'lolo':
        # get labels from same labelfile but from different columns
        # FIXME: make this optional
        _, labels_fold = pc.getFiles(args.inputfolder, args.suffix,
                                     labelfile=args.labelfile, label_col=2)
        le = preprocessing.LabelEncoder()
        labels_fold = le.fit_transform(labels_fold)
    else:
        labels_fold = labels

    # make outer cv
    # create outer-cv with 1 fold if we have a test label
    test_labels = None
    if args.ti:
        test_dir = args.ti if args.ti else args.inputfolder

        test_files, test_labels = pc.getFiles(test_dir, args.suffix,
                                              labelfile=args.tl)
        print 'test_labels:', test_labels
        print 'have {} test labels'.format(len(test_files))
        assert(test_files is not None)
        assert(len(test_files) > 0)
        test_descr = pc.loadDescriptors(test_files)
        checkArray(test_descr, 'test descriptors')
        if args.distances:
            writeDistanceMatrix(test_descr, args.outputfolder,
                            test_files, args.taskname, args.suffix)
    if test_labels is not None or args.outer_nfolds == 1:
        if args.outer_nfolds == 1:
            print 'set one fold -> just use inner-cv'
            descr = np.concatenate([descr, descr], axis=0)
            if test_labels is None:
                outer_cv = [(np.arange(len(labels)),
                             np.arange(len(labels)))]
            else:
                outer_cv = [(np.arange(len(labels)),
                             np.arange(len(labels),len(labels)+len(labels)))]
            labels = np.concatenate([labels, labels])
        else:
            print 'have test labels -> 1 fold'

            descr = np.concatenate([descr, test_descr], axis=0)
            files = np.concatenate([files, test_files], axis=0)

            # deal with strings as labels
            if args.regression:
                test_labels = np.array(test_labels).astype(np.float)
            else:
                bak = test_labels[:]
                print 'test labels:', sorted(set(test_labels))
                test_labels = le.transform(test_labels)
                print 'number of classes:', len(le.classes_)
                print 'classes:', le.classes_
                print 'labels as classes:', le.transform(sorted(set(bak)))

            outer_cv = [(np.arange(len(labels)), # train
                        np.arange(len(labels),len(labels)+len(test_labels)))] # test
            labels = np.concatenate([labels, test_labels])
    else:
        outer_cv = getCV(labels_fold, args.outer_cv, args.outer_nfolds)


    if args.vl:
        assert(args.grid)
        print 'have a validation label file -> use that instead inner cross-val'
        val_dir = args.vi if args.vi else args.inputfolder

        val_files, val_labels = pc.getFiles(val_dir, args.suffix,
                                            labelfile=args.vl)
        # deal with strings as labels
        if args.regression:
            val_labels = np.array(val_labels).astype(np.float)
        else:
            val_labels = le.transform(val_labels)

        val_descr = pc.loadDescriptors(val_files)
        checkArray(val_descr, 'test descriptors')

    all_test = []
    all_pred = []
    all_files = []
    all_dec = []
    fold_score = []
    fold_cls = []

    if le:
        n_classes = len(le.classes_)
    else:
        n_classes = len(set(labels))
    
    if args.load_cls:
        fold_cls = [pc.load(args.load_cls)]
    for e, (train, test) in enumerate(outer_cv):
        print '---\nFold {} / {}'.format(e+1, len(list(outer_cv)))
        print 'train w. {}, test w. {}'.format(len(train), len(test))
        descr_tr = prep.fit_transform(descr[train])
#            sc = preprocessing.StandardScaler()
#            descr_tr = sc.fit_transform(descr[train])
        if not args.load_cls:
            if grid:
                # create inner gridsearch
                if args.vl:
                    # use validation set --> only 1 fold needed
                    val_descr = prep.transform(val_descr)
                    descr_tr = np.concatenate([descr_tr, val_descr], axis=0)
                    inner_cv = [(np.arange(len(labels[train])),
                                np.arange(len(labels[train]),len(labels[train])+len(val_labels)))]
                    labels_tr = np.concatenate([labels[train], val_labels])
                else:
                    print 'inner CV with {}'.format(args.nfolds)
                    inner_cv = getCV(labels_fold[train], args.cv, args.nfolds)

                if args.my_grid_cv:
                    cls_cv = myGridCV(descr_tr, labels[train],
                                      train_sample_weights,
                                      test_sample_weights,
                                      cls, grid, inner_cv,
                                      #average=False,
                                      average=True,
                                      parallel=args.parallel,
                                      nprocs=args.nprocs,
                                      scoring=myScorer(args.scorer, n_classes,
                                          args.regression))

                    # Refit using all the data
                    cls_cv.fit(descr_tr, labels[train],
                            sample_weight=train_sample_weights)
                else:
                    n_jobs = (args.nprocs if args.parallel else 1)

                    cls_cv = GridSearchCV(cls, grid,
                                      # spearman scoring or normal (accuracy/r2)
                                      scoring=myScorer(args.scorer, n_classes),
                                      cv=inner_cv, n_jobs=n_jobs,
                                      fit_params={'sample_weight':
                                          train_sample_weights}
                                      )

                    if args.labelfile2:
                        print 'got more labels -> MultiOutputClassifier'
                        cls_cv = MultiOutputClassifier(cls_cv)

                    if args.proba:
                        cls_cv = CalibratedClassifierCV(cls_cv)

                    # NOTE GridSearchCV does not support sample weights during
                    # scoring.
                    cls_cv.fit(descr_tr, labels[train])

                if not args.labelfile2:
                    print 'fold:', e+1, 'best CLS:', (cls_cv if args.my_grid_cv else\
                                                      cls_cv.best_estimator_)
                    print 'best score:', cls_cv.best_score_
            else:  # no grid
                cls_cv = cls
                if args.labelfile2:
                    cls_cv = MultiOutputClassifier(cls_cv)
                if args.proba:
                    cls_cv = CalibratedClassifierCV(cls_cv)
                # fit
                cls_cv.fit(descr_tr, labels[train],
                        sample_weight=train_sample_weights)

        # evaluate
#            descr_te = sc.transform(descr[test])
        descr_te = prep.transform(descr[test])
        pred = cls_cv.predict(descr_te)

        if args.proba:
            dec = cls_cv.predict_proba(descr_te)
        else:
            try:
                dec = cls_cv.decision_function(descr_te)
            except AttributeError as err:
                print 'Warning: use predict_proba now (error: {})'.format(err)
                try:
                    dec = cls_cv.predict_proba(descr_te)
                except AttributeError:
                    print 'using predict'

                    try:
                        dec = cls_cv.predict(descr_te)
                    except AttributeError:
                        print 'no decision-function / proba'
                        dec = None

        pred = thresholdPredictions(pred, labels[train], args.min_pred,
                             args.max_pred)

        labels_te = labels[test]
        scorer = myScorer(args.scorer, n_classes, args.regression)
        if args.labelfile2:
            t1,t2 = zip(*labels_te)
            p1,p2 = zip(*pred)

            s1 = scorer._score_func(t1, p1, **scorer._kwargs)
            s2 = scorer._score_func(t2, p2, **scorer._kwargs)
            print '(fold) acc1:', s1
            print '(fold) acc2:', s2
            s = (s1 + s2) / 2.0
        else:
            s = scorer._score_func(labels_te, pred,
                    sample_weight=test_sample_weights, **scorer._kwargs)

        fold_score.append(s)
        print len(files), len(test)
        all_files.append(np.array(files)[test])
        all_pred.append(pred)
        all_test.append(labels_te)
#        all_pred.extend(pred.tolist())
#        all_test.extend(labels_te.tolist())
        all_dec.append(dec)
        fold_cls.append(copy.deepcopy(cls_cv))
    # end for-loop
    
    all_files = np.concatenate(all_files, axis=0)
    all_pred = np.concatenate(all_pred, axis=0)
    all_test = np.concatenate(all_test, axis=0)

    if args.regression and not args.absvalue:
        spearman, p =  stats.spearmanr(all_pred, all_test)
        r2 = r2_score(all_test, all_pred)
        rmse = np.sqrt(mean_squared_error(all_test, all_pred))

        errors = []
        for i in range(len(all_test)):
            diff = all_pred[i] - all_test[i]
            errors.append(diff)

        # all errors
        with open(os.path.join(args.outputfolder,'errors.txt'), 'w') as f:
            for e in errors:
                f.write('{}\n'.format(e))

        # errors per date
        with open(os.path.join(args.outputfolder,'date_errors.txt'), 'w') as f:
            for lab,err in zip(all_test,errors):
                f.write('{} {}\n'.format(lab,err))

        # absolute errors
        aes = np.abs(errors)

        # explicitly for dating:
        cs = []
        for i in range(5,51,5):
            cs_i = len(aes[aes <= i])
            cs_i /= float(len(aes))
            cs.append(cs_i)


        # mean absolute errors
        mae = np.mean(aes)
        mae_std = np.std(aes)
        # median absolute errors
        medae = np.median(aes)

        print '---\nspearman: {} p:{}'.format(spearman, p)
        print 'r2: {}\nrmse: {}'.format(r2, rmse)
        print 'mae: {}'.format(mae)
        print 'mae-std: {}'.format(mae_std)
        print 'median ae: {}'.format(medae)
        print 'cs:', cs

        # average score doesnt make sense for leave-one-out
        if args.outer_cv != 'loo' and args.outer_cv != 'lolo':
            print 'WARNING: average fold score for loo/lolo might irritate'
        print ('average fold-score: mean: {}, std: {}, median:'
              ' {}').format(np.mean(fold_score), np.std(fold_score),
                            np.median(fold_score))
        stats_ret = {'r2':r2, 'rmse':rmse, 'mae':mae, 'spear':spearman,
                     'spear-p':p, 'cs':cs, 'medae':medae,
                    'mae-std':mae_std}
        evaluate.write_stats(os.path.join(args.outputfolder,
                                                  'stats_cls'), stats_ret, args.identifier)
        print 'some predictions:', all_pred[:15], all_test[:15]
        if le is not None:
            all_pred = le.inverse_transform(all_pred)

        with open(os.path.join(args.outputfolder, 'predictions.txt'),'w') as f:
            for p in all_pred:
                f.write('{}\n'.format(p))

        decisions = np.concatenate(np.asarray(all_dec))

        if decisions.ndim == 1:
            decisions = np.expand_dims(decisions, axis=1)

        np.savetxt(os.path.join(args.outputfolder, 'decisions.txt'),
                decisions)

    else:
        if args.absvalue:
            all_pred = np.round(all_pred)


#        FIXME average intraclass distance
#        descr_te = descr[test]
#        pred = cls_cv.predict(descr_te)
        #        maid = 0
#        for k,v in aid.iteritems():
#            maid += np.mean(v)
#        print 'AID:', maid / len(aid.keys())

#        aid = {}
#        aid.setdefault(all_test[r], []).append(dec_norm[r,k])

        # Note: micro-averaged pr,re,f1 is the same as acc!
#        pr, re, f1, _ = precision_recall_fscore_support(all_test,
#                                                 all_pred,
#                                                 pos_label=None,
#                                                 average='micro')
#        print 'pr: {}, re: {}, f1: {}'.format(pr, re, f1)

        if args.labelfile2:

#            all_test = le.inverse_transform(all_test)
#            all_pred = le.inverse_transform(all_pred)

            t1,t2 = zip(*all_test)
            p1,p2 = zip(*all_pred)

            acc1 = accuracy_score(t1, p1)
            acc2 = accuracy_score(t2, p2)
            print 'acc1:', acc1
            print 'acc2:', acc2

            stats_ret = cls_stats(t1, p1, all_dec, test_sample_weights)
            stats_ret = cls_stats(t2, p2, all_dec, test_sample_weights)

            # TODO: can we merge it somehow with below?
        else:

            stats_ret = cls_stats(all_test, all_pred, all_dec,
                    test_sample_weights)
#        stats_ret = {'acc':acc,
#                     'kappa':kappa, 'rmac':rmac, 'rmic':rmic, 'pmic':pmic,
#                     'pmac':pmac, 'fmic':fmic, 'fmac':fmac}

        # TODO: use maybe computeStats
#        print 'classifyNN:'
#        _, stats_n = evaluate.classifyNN(None, None, all_test, ???,
#                                         distance=False,
#                                         ret_matrix=all_dec)

            evaluate.write_stats(os.path.join(args.outputfolder,
                                                  'stats_cls'), stats_ret, args.identifier)
            print 'some predictions:', all_pred[:15], all_test[:15]
            if le is not None:
                all_pred = le.inverse_transform(all_pred)
                all_test = le.inverse_transform(all_test)
            with open(os.path.join(args.outputfolder, 'predictions.txt'),'w') as f:
                for p in all_pred:
                    f.write('{}\n'.format(p))
            with open(os.path.join(args.outputfolder, 'prediction_files.txt'),'w') as f:
                def longest_common_suffix(list_of_strings):
		    reversed_strings = [s[::-1] for s in list_of_strings]
		    reversed_lcs = os.path.commonprefix(reversed_strings)
		    lcs = reversed_lcs[::-1]
		    return lcs 
    
                print 'all_files[0]', all_files[0]
                suf = longest_common_suffix(all_files)
                for e,p in enumerate(all_files):
                    fout = os.path.basename(p)
                    fout = fout.replace(suf, '')
                    #f.write('{} {}\n'.format(fout,all_test[e]))
                    f.write('{} {} {}\n'.format(fout,all_test[e], all_test[e] ==
                                               all_pred[e]))

            decisions = np.concatenate(np.asarray(all_dec))

            if decisions.ndim == 1:
                decisions = np.expand_dims(decisions, axis=1)

            np.savetxt(os.path.join(args.outputfolder, 'decisions.txt'),
                    decisions)

    # only predictions (args.tl is None)
    if args.ti and test_labels is None:
        if len(fold_cls) > 1:
            print fold_score
    #        print fold_cls
            ind = np.argmax(fold_score)
            print 'use score index {}'.format(ind)

            cls = fold_cls[ind]
        else:
            cls = fold_cls[0]
        # refit -> assurance that we really did that
        if not args.load_cls:
            cls.fit(descr, labels) #sample_weight=train_sample_weights)
        pred = cls.predict(test_descr)
        if le:
            print 'le inverse transform'
            print 'le classes:', le.classes_
            pred_labels = le.inverse_transform(pred)
#            n_classes = len(le.classes_)
        else:
            pred_labels = pred
#            n_classes = len(set(pred))

        if args.proba:
            print 'predict probas'
            dec = cls_cv.predict_proba(test_descr)
        else:
            try:
                dec = cls.decision_function(test_descr)
            except AttributeError as err:
                print 'Warning: use predict_proba now (error: {})'.format(err)
                try:
                    dec = cls_cv.predict_proba(test_descr)
                except:
                    print 'no decision-function / proba'
                    dec = None
        if dec is not None:
            dec = preprocessing.normalize(dec, norm='l1')
        assert(len(test_files) == len(dec))
        if le:
            assert(dec.shape[1] == len(le.classes_))

        print '> write belonging stuff'
        writeBelonging(pred_labels, dec, args.outputfolder, test_files,
                       args.taskname, args.suffix, le)
        pc.dump(os.path.join(args.outputfolder, 'cls'), cls)

    return all_dec, stats_ret

    # final evaluation if we have had an inner cross-validation
    # FIXME: only use if you want to choose the best cls
    #        to use with another independent test set
    """
    if grid:
        # choose best model upon:
        # a) mode (most used params of grid
        if args.refit == 'mode':
            # restructure
            cls_params = {}
            for cls_f in fold_cls:
                for k,v in cls_f.get_params().iteritems():
                    cls_params.setdefault(k, []).append(v)
            final_params = {}
            for k,v in cls_params.iteritems():
                final_params[k] = stats.mode(v)[0].item()
            cls.set_params(**final_params)

        # b) best score
        elif args.refit == 'score':
            if scorer == 'mse':
                ind = np.argmin(fold_score)
            else:
                ind = np.argmax(fold_score)
            cls = fold_cls[ind]

        else:
            raise ValueError('unknown refit method')

        print 'refit now w. cls:', cls
        print ('evaluate full cross-validation again...'
               ' -> this is actually cheating!')
        all_pred = []
        for e, (train, test) in enumerate(outer_cv):
            descr_tr = prep.fit_transform(descr[train])
#                sc = preprocessing.StandardScaler()
#                descr_tr = sc.fit_transform(descr[train])
            cls.fit(descr_tr, labels[train])

            descr_te = prep.transform(descr[test])
#                descr_te = sc.transform(descr[test])
            pred = cls.predict(descr_te)
            thresholdPredictions(pred, labels[train], args.min_pred,
                                 args.max_pred)

            all_pred.extend(pred.tolist())
    all_pred = np.array(all_pred)

    if args.regression:
        spearman, p =  stats.spearmanr(all_pred, all_test)
        r2 = r2_score(all_test, all_pred)
        mse = np.sqrt(mean_squared_error(all_test, all_pred))

        print 'REFIT---\nspearman: {} p:{}'.format(spearman, p)
        print 'r2: {}\nrmse: {}\n'.format(r2, mse)
    else:
        print 'score:', accuracy_score(all_test, all_pred)
        p, r, f, _ = precision_recall_fscore_support(all_test,
                                                 all_pred,
                                                 average='micro')
        print p, r, f
        cm = confusion_matrix(all_test, all_pred)
        print 'map:', MAP(all_test, all_pred)

#        print cm
#        print le.inverse_transform(np.arange(len(le.classes_)))
        print (classification_report(pred_labels, t_labels))


    print 'some predictions:', all_pred[:15], all_test[:15]
    """

if __name__ == '__main__':
    """
    classify with train labels and test on test labels
    """

    prep = preprocess.Preprocess.fromArgs()
    parser = argparse.ArgumentParser('classify',
                                    parents=[prep.parser])

    parser = pc.commonArguments(parser)
    parser = parseArguments(parser)
    parser.add_argument('--ex', action='store_true',
                        help='exemplar classification')
    args = parser.parse_args()

    pc.mkdir_p(args.outputfolder)
    if args.log:
        log = pc.Log(sys.argv, args.outputfolder)

    if args.outputfolder and not os.path.exists(args.outputfolder):
        pc.mkdir_p(args.outputfolder)

    run(args)

    sys.exit()

# FIXME: fix code below

    if args.ex:
#        X_train, X_test, y_train, y_test = cross_validation.train_test_split(descr, labels,
#                                                            test_size=0.5,
#                                                            stratify=labels)
#        ex_cls = exemplar_cls_quwi.computeExCls(X_train, cls, len(X_train),
#                              outputfolder=args.outputfolder,
#                              labels=y_train, parallel=args.parallel,
#                              nprocs=args.nprocs, use_labels=True, files=files)
         ex_cls = exemplar_cls_quwi.computeExCls(descr, cls, len(descr),
                              outputfolder=args.outputfolder,
                              labels=labels, parallel=args.parallel,
                              nprocs=args.nprocs, use_labels=True, files=files)

#        ab_list = exemplar_cls_quwi.computeAB(X_test, ex_cls, y_test)

    else:
        cls.fit(descr, labels)
#    print 'best est:', cls.best_estimator_

    pc.dump(args.clsname, cls)
    t_files, t_labels = pc.getFiles(args.ti, args.suffix,
                                    labelfile=args.tl)
#    new_labels = []
#    for l in t_labels:
#        new_labels.append( l if len(l) == 2 else '0'+l)
#    t_labels = new_labels
    t_labels = le.transform(np.array(t_labels))
    t_descr = pc.loadDescriptors(t_files)
#    t_descr = preprocessing.normalize(t_descr)
#    t_descr = sc.transform(t_descr)

    if args.ex:
        pred_labels = []
#        for des in t_descr:
#            cl = np.zeros( (len(np.unique(labels))), dtype=np.uint32)
#            for i, ex in enumerate(ex_cls):
#                cl[ labels[i] ] += ex.predict(des.reshape(1,-1))
#            pred_labels.append( cl.argmax() )
        all_scores = []
#        for i in range(len(t_descr)):
        df = exemplar_cls.predictExemplarCls(t_descr, ex_cls)
#        df = np.maximum(df, -1.0) # ?
#        df = exemplar_cls_quwi.convertToProbs(df, ab_list)
        sc = np.argmax(df, axis=1)

        pred_labels = []
        for i in range(len(t_descr)):
            #pred_labels.append( y_train[ sc[i] ] )
            pred_labels.append( labels[ sc[i] ] )
        pred_labels = np.array(pred_labels)
#        sc = exemplar_cls.voteCls(df)
#        all_scores.append(sc)
#        scores = np.concatenate(all_scores, axis=0)
#        pred_labels = np.argmax(scores, axis=1)
    else:
        pred_labels = cls.predict(t_descr)



#    cm = confusion_matrix(t_labels, pred_labels)
#    print 'map:', MAP(t_labels, pred_labels)

#    print cm
#    print le.inverse_transform(np.arange(len(le.classes_)))
#    print (classification_report(pred_labels, t_labels))
    if args.log:
        log.dump()
