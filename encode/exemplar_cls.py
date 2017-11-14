#!/usr/bin/python2
import puhma_common as pc
import evaluate
import numpy as np
import os
import argparse
import gzip
import cPickle
import classification
import progressbar
from sklearn.linear_model import SGDClassifier
from sklearn.svm import OneClassSVM, LinearSVC
from sklearn import cross_validation
from sklearn import decomposition
from sklearn.model_selection import GroupKFold 
from sklearn.datasets import dump_svmlight_file
from sklearn import preprocessing
from sklearn.grid_search import ParameterGrid, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report,\
                           recall_score, roc_auc_score, f1_score,\
                            precision_recall_fscore_support,\
                            roc_curve
from sklearn.multiclass import OneVsRestClassifier
import copy
from scipy import linalg
import sys
import preprocess
import evaluate
import exemplar_cls_quwi

def crossVal(estimator, grid, neg_descr, bg_labels, parallel, nprocs):
    # FIXME: use lklo or strat cv (depending on situation)
    le = preprocessing.LabelEncoder()
    bg_labels = le.fit_transform(bg_labels)
    nfolds = 2
    gkf = GroupKFold(n_splits=nfolds)
    best_cls = None
    best_map = 0.0
    all_combos = list(ParameterGrid(grid))
    print 'lets do a cross-evaluation of all cls'
    for i in range(len(all_combos)):
        current_map = 0.0
        for train, test in gkf.split(neg_descr, bg_labels, groups=bg_labels):            
            cls = copy.deepcopy(estimator)
            cls.set_params(**all_combos[i])
            print '\ttest cls:', cls
            widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
                   progressbar.ETA()]
            progress = progressbar.ProgressBar(widgets=widgets,
                                       maxval=len(neg_descr))
            ex_cls = exemplar_cls_quwi.computeIndependentExCls(neg_descr[test], 
                                                               neg_descr[train],
                                                               cls,
                                                               parallel=parallel, 
                                                               nprocs=nprocs)
            """
            def computeEx(k):
                neg = neg_descr[ bg_labels != bg_labels[k] ]
                ex_c = createExemplarCls(neg_descr[k].reshape(1,-1),
                    neg,
                    cls)
                progress.update(k+1)
                return ex_c
            progress.start()
            if parallel:
                ex_cls = pc.parmap(computeEx, range(len(neg_descr)), nprocs=nprocs)
            else:
                ex_cls = map(computeEx, range(len(neg_descr)))
            progress.finish()
            """
        
            scores_mat = predictExemplarCls(neg_descr[test], ex_cls)
            if isinstance(ex_cls[0], OneClassSVM):
                np.fill_diagonal(scores_mat, np.finfo(float).max)
                stats = evaluate.computeStats('sum/max', scores_mat, 
                                          bg_labels[test], 
                                          distance=True, parallel=parallel)  
            else:
                np.fill_diagonal(scores_mat, np.finfo(float).min)
                stats = evaluate.computeStats('sum/max', scores_mat, 
                                          bg_labels[test], 
                                          distance=False, parallel=parallel)  
            current_map += stats['mAP']

        current_map /= nfolds
        if current_map > best_map:
            best_map = current_map
            best_cls = cls
    best_cls
    print 'cross val has chosen: ', best_cls
    return best_cls

def createExemplarCls(ex_desc, neg_desc,
                      the_cls, resampling=0):
    if isinstance(the_cls, OneClassSVM):
        cls = OneClassSVM(kernel='linear', nu=1.0/len(neg_desc))
    else:
        cls = copy.deepcopy(the_cls)

    
    if resampling > 0:
        cur_labels = []
        num_pos = neg_desc.shape[0] /  resampling
        cur_data = np.zeros( (neg_desc.shape[0] + num_pos, neg_desc.shape[1]),
                            neg_desc.dtype )
        indices =\
            np.random.permutation(np.arange(len(neg_desc)))

        cnt = 0
        for i in range(len(neg_desc)+num_pos):
            if i  % resampling == 0:
                cur_labels.append(1)
                cur_data[i] = ex_desc
            else:
                cur_labels.append(0)
                cur_data[i] = neg_desc[indices[cnt]]
                cnt = cnt + 1
#        cur_data = np.concatenate(cur_data, axis=0) 
#        print('createExemplarCls: cur_data.shape: {}'.format(cur_data.shape))
    else:
        cur_labels = [1] * ex_desc.shape[0]
        cur_labels.extend( [0] * neg_desc.shape[0] )
        if isinstance(the_cls, OneClassSVM):
            #cur_data = neg_desc - ex_desc
            cur_data = neg_desc
        else:
            cur_data = np.concatenate([ex_desc, neg_desc], axis=0)

    if np.isnan(cur_data).any():
        print 'WARNING have a nan in the cur_data (createExCLS)'
    if np.isinf(cur_data).any():
        print 'WARNING have a inf in the cur_data (createExCLS)'

    if isinstance(the_cls, OneClassSVM):
        cls.fit(cur_data)
        #cls.coef_new = (ex_desc - cls.coef_).reshape(cls.coef_.shape)
        cls.coef_new = (ex_desc -\
                        np.sum(cls.support_vectors_ *
                               cls.dual_coef_.T,axis=0)).reshape(cls.coef_.shape)
    else:
        cls.fit(cur_data, cur_labels)

    return cls

def createExemplarClsFromFile(ex_file, b_files, 
                              cls, clsname='sgd',
                              subfolds=1,
                              average=False, 
                              weights=(0.5,0.01) ):
    """
    parameters:
        ex_descr: descriptor(s) for which to make an exemplar-classifier
        b_files: files containing the negative descriptors
        cls: the classifier base class
    returns: the exemplar classifier
    """

    # load descriptors to compute an exemplar classifier for
    # == the positive class
    ex_desc = pc.loadDescriptors(ex_file)
    if average:
        ex_desc = np.mean(ex_desc, axis=0).reshape(1,-1)

    if clsname == 'sgd' and subfolds > 1:
        file_groups = np.array_split(b_files, subfolds)
        ex_desc_splits = np.array_split(ex_desc, subfolds)       
    elif average:
        file_groups = [ b_files ]
        ex_desc_splits = [ ex_desc ]

    if (clsname == 'sgd' and subfolds > 1) or average:
        cls = copy.deepcopy(cls)
        # training part
        for e, cur_files in enumerate(file_groups):
            cur_data = [ ex_desc_splits[e] ]
            cur_labels = [1] * ex_desc_splits[e].shape[0]
            
            # insert negatives from background files
            for f in range(len(cur_files)):
                temp_data = pc.loadDescriptors(cur_files[f])
                if temp_data == None:
                    print 'couldnt load', f
                    continue
                if args.average:
                    temp_data = np.mean(temp_data,axis=0).reshape(1,-1)
                cur_data.append(temp_data)
                cur_labels.extend( [0]*temp_data.shape[0] )

            cur_data = np.concatenate(cur_data, axis=0)
            
            sample_weight = [weights[0]] * ex_desc.shape[0]
            sample_weight.extend( [weights[1]] * (len(cur_labels)-ex_desc.shape[0]) )
            if args.clsname == 'sgd':
                cls.partial_fit(cur_data, cur_labels, classes=[1,0],
                                sample_weight=sample_weight)
            else:                
                cls.fit(cur_data, cur_labels, sample_weight=sample_weight)
            
            del cur_data, cur_labels
    # faster process:            
    else:
        neg_desc = pc.loadDescriptors(b_files)
        createExemplarCls(ex_desc, neg_desc, cls, weights)

    return cls

def predictLoadECLS(descr_probe, folder, files, suffix='_ecls.pkl.gz',
                    parallel=False, nprocs=None):
    print '=> predict by loading E-CLS'
    if np.isnan(descr_probe).any():
        print 'WARNING have a nan in the descr_probe'
    if np.isinf(descr_probe).any():
        print 'WARNING have a inf in the descr_probe'

    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
               progressbar.ETA()]
    progress = progressbar.ProgressBar(widgets=widgets, 
                                       maxval=len(files) )
    def compute(i):
        if files[i].endswith('.pkl.gz'):
            fname = files[i].replace('.pkl.gz', suffix)
        else:
            fname = os.path.splitext(files[i])[0] + suffix
        fname = os.path.join(folder, os.path.basename(fname))
        cls = pc.load(fname)
        if isinstance(cls, OneClassSVM):

            coef = cls.coef_new
            mag = np.sqrt(coef.flatten().dot(coef.flatten()))
#            sc = descr_probe.dot( (coef / mag).reshape(-1,1)) +\
#                                 (cls.intercept_ / mag).reshape(-1,1)
            sc = descr_probe.dot( (coef / mag).reshape(1,-1) )
        else:
            sc = cls.decision_function(descr_probe).reshape(1,-1)
        if np.isnan(sc).any():
            print 'WARNING have a nan in sc'
        if np.isinf(sc).any():
            print 'WARNING have a inf in sc'

        if sc.shape[1] != descr_probe.shape[0]:
            print '{}x{} dot {}x{}'.format(descr_probe.shape[0], 
                                           descr_probe.shape[1],
                                           cls.coef_.shape[0],
                                           cls.coef_.shape[1])
            raise ValueError('sc.shape[0] {} != descr.probe.shape[0]'
                             ' {}'.format(sc.shape[0], descr_probe.shape[0]))
        progress.update(i+1)
        return sc

    progress.start()
    if parallel:
        score = pc.parmap( compute, range(len(files)), nprocs=nprocs )
    else:
        score = map(compute, range(len(files)))
    all_scores = np.concatenate(score, axis=0)
    progress.finish()

    return all_scores


def predictExemplarCls(ex_desc, ex_cls):
    score = []
    for cl in ex_cls:
#        if isinstance(cl, OneClassSVM):
#            sc = ex_desc.dot(cl.coef_.reshape(-1,1)) -\
#                                 cl.intercept_.reshape(-1,1)
#        else:
        try:
            sc = cl.decision_function(ex_desc)
        except AttributeError:
            sc = cl.predict_proba(ex_desc)[:,1]
        score.append(sc.reshape(-1,1))
    
    # TODO: maybe add here platt-normalization
    # --> only useful for multiple instances per class
    all_scores = np.concatenate(score, axis=1)
    return all_scores

def voteCls(all_scores, vote='sum'):
    # majority-vote
    if vote == 'majority':
        # search maximum for each sample
        ind = np.argmax(all_scores, axis=1)
        v = np.bincount(ind, minlength=len(ex_cls)).reshape(1,-1)
    # or sum-vote
    elif vote == 'sum':
        v = np.sum(all_scores, axis=0).reshape(1,-1)
    elif vote == 'max':
        return np.argmax(all_scores, axis=1)
    else:
        raise ValueError('vote-method ({}) unknown'.format(vote))

    return v

def addArguments(parser):
    parser.add_argument('--average', action='store_true',
                        help='average descriptors of each file')
    parser.add_argument('--bl', '--bg_labelfile', 
                        help='background-labelfile')
    parser.add_argument('--bi', '--bg_inputfolder',\
                        help='the background input folder of the features')
    parser.add_argument('--bs', '--bg_suffix', default='',\
                        help='only chose those with a specific suffix')
    parser.add_argument('--load_bg', 
                        help='load background classification stuff')
    parser.add_argument('--load_mean', help='load mean file')
    parser.add_argument('--load_cov', help='load covariance file')
    parser.add_argument('--load_weights', help='load weights')
    parser.add_argument('--load_scores', 
                        help='load scores')
    parser.add_argument('--lda_like', action='store_true',
                        help='perform lda-like computations')
    return parser

if __name__ == '__main__':
#    import stacktracer
#    stacktracer.trace_start("trace.html",interval=5,auto=True)
    
    parser = argparse.ArgumentParser('classify encodings')
    parser = preprocess.parseArguments(parser) 
    parser = pc.commonArguments(parser)
    parser = classification.parseArguments(parser)
    parser = addArguments(parser)
    args = parser.parse_args()

    if not os.path.exists(args.outputfolder):
        pc.mkdir_p(args.outputfolder)

    if not args.scale and not args.load_scaler and\
       ('svm' in args.clsname or args.clsname == 'sgd'):
        print 'WARNING: svm or sgd chosen but not --scale!' 

    all_cls = args.func(args)
    if not all_cls:
        print 'no classifier given'


    files, labels = pc.getFiles(args.inputfolder, args.suffix,
                                labelfile=args.labelfile)
    files = np.array(files)
    labels = np.array(labels)

    # these are our background / negative training files
    b_files, b_labels = pc.getFiles(args.bi, args.bs if args.bs else
                                    args.suffix, labelfile=args.bl)

    # let's first test shapes
    test_f = pc.loadDescriptors(files[0])
    b_test_f = pc.loadDescriptors(b_files[0])
    assert(test_f.shape[1] == b_test_f.shape[1])
    print 'descriptor-dimension:', test_f.shape[1]

    # let's shuffle them
    shuffle_ids = np.arange(len(b_files))
    np.random.shuffle(shuffle_ids)
    b_files = np.array(b_files)[shuffle_ids]
    b_labels = np.array(b_labels)[shuffle_ids]

    prep = preprocess.Preprocess(\
                                 pca_components = args.pca_components,
                                 normalize=args.normalize\
                                )
                          
    if args.load_pca:
        prep.load_pca(args.load_pca)
 
    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
                   progressbar.ETA()]
  



    # currently support only one classifier
    the_cls = all_cls[0]

    if args.aug:
        prep.strip_aug = args.aug
        prep.aug = None

    def exemplar_classify(i):
        cls = copy.deepcopy(the_cls)
        # load descriptors to compute an exemplar classifier for
        # == the positive class
        if args.pq:
            ex_desc = prep.uncompress(pos_desc[i])
        else:
            ex_desc = pc.loadDescriptors( files[i] )
        if args.average:
            ex_desc = np.mean(desc, axis=0).reshape(1,-1)

        if args.clsname == 'sgd' and args.subfolds > 1:
            file_groups = np.array_split(b_files, args.subfolds)
            ex_desc_splits = np.array_split(ex_desc, args.subfolds)
        else:
            file_groups = [b_files]
            ex_desc_splits = [ ex_desc ]

        # training part
        for e, cur_files in enumerate(file_groups):
            cur_data = [ ex_desc_splits[e] ]
            cur_labels = [1] * ex_desc_splits[e].shape[0]
            
            # insert negatives from background files
            for f in range(len(cur_files)):
                if args.pq:
                    temp_data = prep.uncompress(neg_desc[f])
                else:
                    temp_data = pc.loadDescriptors(cur_files[f])
#                                                   max_descs_per_file=max_descs)
                    if temp_data == None:
                        print 'couldnt load', f
                        continue
                if args.average:
                    temp_data = np.mean(temp_data,axis=0).reshape(1,-1)
                cur_data.append(temp_data)
                cur_labels.extend( [0]*temp_data.shape[0] )

            cur_data = np.concatenate(cur_data, axis=0)
#            print 'cur_data', cur_data.shape, cur_data.dtype
            
            cur_data = prep.transform(cur_data)

            sample_weight = [0.5] * ex_desc.shape[0]
            sample_weight.extend( [0.01] * (len(cur_labels)-ex_desc.shape[0]) )
            if args.clsname == 'sgd':
                cls.partial_fit(cur_data, cur_labels, classes=[0,1],
                                sample_weight=sample_weight)
            else:                
                cls.fit(cur_data, cur_labels, sample_weight=sample_weight)
            
            del cur_data, cur_labels

#                filename = os.path.join(args.outputfolder, args.clsname) +'.pkl.gz'
#                with gzip.open(filename, 'wb') as fOut:
#                    cPickle.dump(cls, fOut, -1)
#                    print 'saved', filename
        progress.update(i+1)        
        return cls

    filename = os.path.join(args.outputfolder, args.clsname + '_all.pkl.gz')
    if args.load_cls:
        with gzip.open(filename, 'rb') as f:
            ex_cls = cPickle.load(f)
            print 'loaded', filename
    else:
        progress = progressbar.ProgressBar(widgets=widgets, maxval=len(files))
        progress.start()
        if args.parallel:
            ex_cls = pc.parmap(exemplar_classify, range(len(files)), nprocs=args.nprocs)
        else:
            ex_cls = map(exemplar_classify, range(len(files)))
        progress.finish()

        pc.dump(filename, ex_cls)


    print 'progress predict'

    # iteratively predict
    def multi_predict(i):
        if args.pq:
            ex_desc = prep.uncompress(pos_desc[i])
        else:
            ex_desc = pc.loadDescriptors(files[i])
        ex_desc = prep.transform(ex_desc)
        score = []
        for e,cl in enumerate(ex_cls):
            if e == i:
                sc = np.zeros( ex_desc.shape[0] )
            else:
                sc = cl.decision_function(ex_desc)
                # TODO: maybe add here platt-normalization
            score.append(sc.reshape(1,-1))
        
        all_scores = np.concatenate(score, axis=0)
        
        # search maximum for each sample
        ind = np.argmax(all_scores, axis=0)
        # majority-vote
        vote = np.bincount(ind, minlength=len(ex_cls)).reshape(1,-1)

        # or sum-vote
        sumi = np.sum(all_scores, axis=1).reshape(1,-1)
        
        progress.update(i+1)
        return vote, sumi
#        return np.array(score).reshape(1,-1)

    filename = os.path.join(args.outputfolder, 'scores.pkl.gz')
    if args.load_scores:
        with gzip.open(filename, 'rb') as f:
            all_scores = cPickle.load(f)
            print 'loaded', filename
    else:
        progress = progressbar.ProgressBar(widgets=widgets, maxval=len(files))
        progress.start()
        if args.parallel:
            scores_vote, scores_sum = zip(*pc.parmap(multi_predict,
                                                     range(len(files)),
                                                     nprocs=args.nprocs))
        else:
            scores_vote, scores_sum = zip(*map(multi_predict,
                                               range(len(files))))
        all_scores = (scores_vote, scores_sum)
        progress.finish()
#        all_scores = np.concatenate(all_scores, axis=0)

        pc.dump(filename, all_scores)
    
    s_vote, s_sum = all_scores
    
    s_vote = np.concatenate(s_vote, axis=0)
    s_sum = np.concatenate(s_sum, axis=0)

#    print len(files), all_scores.shape
    np.fill_diagonal(s_vote, np.finfo(float).min)
    evaluate.computeStats('vote', s_vote, labels, parallel=args.parallel, distance=False) 

    np.fill_diagonal(s_sum, np.finfo(float).min)
    stats = evaluate.computeStats('sum', s_sum, labels, parallel=args.parallel, distance=False) 
    print 'not-correct-pair-ids:'
    for anc in stats['top1_fail']:
        for nc in anc:
            print '{} {}'.format(labels[nc[0]], labels[nc[1]])
    
