import sys
import os
import argparse
import progressbar
import numpy as np
from sklearn import preprocessing
import copy
#import pyvole
import puhma_common as pc
import preprocess
import classification
import exemplar_cls
import evaluate
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn import covariance
from sklearn.svm import OneClassSVM

"""
exemplar classifier:
example call:
    python2 exemplar_cls_quwi.py -i
    /disks/data1/christlein/experiments/pipeline_quwi/1B_Training_rpca/ -l
    labels/quwi_labels_1B_Training.txt --bi
    /disks/data1/christlein/experiments/pipeline_quwi/Training_enc_rpca --bl
    labels/quwi_labels_training_all.txt  --suffix
    _edge_zernike_vlad_wmc_l2g_l2c_pr.pkl.gz --pl
    labels/quwi_labels_1B_Validation.txt --pi
    /disks/data1/christlein/experiments/pipeline_quwi/1B_Validation_rpca/ --tl
    labels/quwi_labels_1B_Test.txt --ti
    /disks/data1/christlein/experiments/pipeline_quwi/1B_Test_rpca/  -o
    quwi_ex_cls  --zeromean --class_weight 1:0.5,0:0.01 --parallel lsvm --C 100
"""

def featureEnc(descr, neg_descr, cls, n_iter, outputfolder, 
               neg_labels, suffix, parallel, nprocs, resampling):

    if isinstance(cls, OneClassSVM):
        cls = OneClassSVM(kernel='linear', nu = 1.0 / len(neg_descr))
        cls.fit(neg_descr)
        tmp = np.sum(cls.support_vectors_ *
                     cls.dual_coef_.T,axis=0).reshape(cls.coef_.shape)
        descr -= tmp
        descr = preprocessing.normalize(descr)
        return descr, None

    
    for n in range(n_iter):
        print 'iter:', n
        ex_cls = computeIndependentExCls(descr, 
                                         neg_descr,
                                         cls,
                                         outputfolder=outputfolder,
                                         suffix=suffix,
                                         parallel=parallel, 
                                         nprocs=nprocs,
                                         resampling=resampling) 
        print '> finished ex-cls computation'
        ex_features = []
        for ex_cl in ex_cls:
            ft = preprocessing.normalize(ex_cl.coef_)
            ex_features.append(ft)
        descr = np.concatenate(ex_features,axis=0)
        print '> finished prepr.'

        # if not last iteration we need to get e-svms for the negative set 
        # TODO: for the not_independent case we should do it as well 
        neg_ex_features = []
        if n < n_iter-1:
            ex_cls_neg = computeExCls(neg_descr,
                                      cls,
                                      len(neg_descr),
                                      labels=neg_labels,
                                      use_labels=True,
                                      parallel=parallel,
                                      nprocs=nprocs)
            for ex_cl in ex_cls_neg:
                ft = preprocessing.normalize(ex_cl.coef_)
                neg_ex_features.append(ft)
            neg_descr = np.concatenate(neg_ex_features, axis=0)
    return descr, neg_descr

def predict(files_probe, ex_cls, prep=None, 
            ex_cls_bg=None, parallel=False, nprocs=None):
    print '| evaluate all E-cls (predict)'
    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
               progressbar.ETA()]
    progress = progressbar.ProgressBar(widgets=widgets, 
                                       maxval=len(files_probe) )
    def predictProbe(i):
        probe_desc = pc.loadDescriptors(files_probe[i])
        if prep:
            if i == 0:
                print 'pre descr[0]', probe_desc[0]
            probe_desc = prep.transform(probe_desc)
            if i == 0:
                print 'post descr[0]', probe_desc[0]
        if ex_cls_bg: # then use cls as attributes
            probe_desc = exemplar_cls.predictExemplarCls(probe_desc, ex_cls_bg)
#            probe_desc = convertToProbs(probe_desc, ab_list)
        df = exemplar_cls.predictExemplarCls(probe_desc, ex_cls)
#        df = convertToProbs(df, ab_list) 
#        df = exemplar_cls.voteCls(df)
        progress.update(i+1)
        return df
        
    progress.start()
    if parallel:
        scores = pc.parmap(predictProbe, range(len(files_probe)), nprocs=nprocs)
    else:
        scores = map(predictProbe, range(len(files_probe)))
    progress.finish()

    scores = np.concatenate(scores, axis=0)
    
    print '[Done]'
    return scores

def computeIndependentExCls(descr, neg_desc, the_cls, 
                            outputfolder=None,
                            suffix='_ecls.pkl.gz',
                            parallel=True, nprocs=None,
                            resampling=0,
                            files=None,
                            load=False,
                            return_none=False,
                            n_cls=-1):

    """
    compute for each descr an exemplar classifier using the descr. of 
    <neg_desc> as negatives, optionally save the classifiers
    """
    print '=> compute independent e-cls'
    if files is not None: assert(len(files) == len(descr)) 
    print outputfolder, len(files) if files else '', suffix, load
    
    if isinstance(the_cls, LDA):
        fname = os.path.join(outputfolder, 'covinv.pkl.gz')
        if load and os.path.exists(fname):
            cov_inv = pc.load(fname)
        else:
#            cc = covariance.GraphLassoCV()
            cc = covariance.ShrunkCovariance()
#            cc = covariance.LeoditWolf()
#            cc = covariance.OAS()
#            cc = covariance.MinCovDet()
            cc.fit(neg_desc)
            cov_inv = cc.precision_

#            covar = np.cov(neg_desc.T, bias=1)
#            # regularize
#            covar[np.diag_indices(len(covar))] += 0.01
#            cov_inv = np.linalg.inv(covar)
            pc.dump(fname, cov_inv, verbose=False)
        print '| elda: cov_inv.shape:', cov_inv.shape
        mean = np.mean(neg_desc, axis=0)
        zero_mean = descr - mean

    if n_cls is not None and n_cls > 0:
        indices = np.random.choice(len(neg_desc), 
                                   min(len(neg_desc),
                                       n_cls),
                                   replace=False)
        neg_desc = neg_desc[indices] 
        print 'choose to use {} neg-descr'.format(len(neg_desc))

    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
               progressbar.ETA()]
    progress = progressbar.ProgressBar(widgets=widgets, maxval=len(descr) )
    def createEx(i):
#        print 'all.shape:', descr.shape, 'one:', descr[i].shape
        fname = ''
        if outputfolder is not None and files is not None:
            if files[i].endswith('.pkl.gz'):
                fname = files[i].replace('.pkl.gz', suffix)
            else:
                fname = os.path.splitext(files[i])[0] + suffix
            fname = os.path.join(outputfolder, os.path.basename(fname))

        if load and fname != '' and os.path.exists(fname):
            run = False
            try:
                cls = pc.load(fname)
                assert(cls.__class__.__name__ == the_cls.__class__.__name__)
                progress.update(i+1)
                if return_none: return None
                return cls
            except: # e.g. EOFError most of the time
                print 'Warning: couldnt load {} -> recompute'.format(fname)
        
#        print 'compute cls for', os.path.basename(files[i])

        if isinstance(the_cls, LDA):
            cls = copy.deepcopy(the_cls)
            w = cov_inv.dot(zero_mean[i].T)
            cls.coef_ = w.reshape(1,-1)
            cls.intercept_ = 0 #np.zeros( (cls.coef_.shape[0],1) )
        else:
            cls = exemplar_cls.createExemplarCls(descr[i].reshape(1,-1),
                                                 neg_desc,
                                                 the_cls, 
                                                 resampling)
        if fname != '':
            pc.dump(fname, cls, verbose=False)
        progress.update(i+1)
        if return_none: return None
        return cls

    progress.start()
    if parallel:
        ex_cls = pc.parmap( createEx, range(len(descr)), nprocs=nprocs )
    else:
        ex_cls = map(createEx, range(len(descr)))
    progress.finish()

    print '[Done]'
    return ex_cls


def computeExCls(descr, the_cls, n_cls,                  
                 outputfolder=None, 
                 labels=None, 
                 suffix='_ecls.pkl.gz',
                 parallel=True, 
                 nprocs=None,
                 use_labels=False,
                 files=None,
                load=False,
                return_none=False):
    if use_labels:
        assert(labels is not None)
        assert(len(descr) == len(labels))
    labels = np.array(labels) # make sure we have a numpy array
    print 'computeExCls: shape', descr.shape, 'take ', n_cls
    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
               progressbar.ETA()]
    progress = progressbar.ProgressBar(widgets=widgets, maxval=n_cls )    
    def createEx(i):
        if use_labels:
            neg = descr[ labels != labels[i] ] 
        else:
            neg = descr[np.arange(len(descr))!=i],        

        fname = ''
        if outputfolder is not None and files is not None:
            if files[i].endswith('.pkl.gz'):
                fname = files[i].replace('.pkl.gz', suffix)
            else:
                fname = os.path.splitext(files[i])[0] + suffix
            fname = os.path.join(outputfolder, os.path.basename(fname))
        if load and fname != '' and os.path.exists(fname):
            cls = pc.load(fname)
            progress.update(i+1)
            if return_none: return None
            return cls
            
        cls = exemplar_cls.createExemplarCls(descr[i].reshape(1,-1),
                                             neg,
                                             the_cls)
        if fname != '':
            pc.dump(fname, cls, verbose=False)
        progress.update(i+1)
        if return_none: return None
        return cls

    progress.start()
    if parallel:
        ex_cls = pc.parmap(createEx, range(n_cls), nprocs=nprocs)
    else:
        ex_cls = map(createEx, range(n_cls))
    progress.finish()

    return ex_cls


def parserArguments(parser):
    parser.add_argument('--attribute', action='store_true',
                        help='use classifiers as attributes')

    parser.add_argument('--pl', '--labelfile_probe', 
                        help='use extra query labels, the other labels target'
                        ' is for the gallery')
    parser.add_argument('--pi', '--inputfolder_probe', 
                        help='give inputfolder for probes')

    parser.add_argument('--bl', '--bg_labelfile',
                        default=[], nargs='*', 
                        help='background-labelfile use this as negatives for'
                        ' exemplar svms or to create attributes')
    parser.add_argument('--bi', '--bg_inputfolder',\
                        default=[], nargs='*',
                        help='the background input folder of the features')
    parser.add_argument('--tl',
                        help='test labels')
    parser.add_argument('--ti',
                        help='test inputfolder')

    parser.add_argument('--load_ex_cls',
                        help='path to exemplar classifiers')
    return parser

def run(args, prep, write_stats=False):
    # create (or load) for each file an exemplar classifier
    # using the rest of the files as background class
    files, labels = pc.getFiles(args.inputfolder, args.suffix,
                                labelfile=args.labelfile)
    # all labels should differ!
    assert( len(set(labels)) == len(labels) )

    # if we use classifiers as attributes then we need
    # background-classifiers independent from the training set
    if args.attribute: assert(args.bi)

    # additional background descriptors
    if len(args.bi) > 0:
        assert(len(args.bi) == len(args.bl))
        bg_files, bg_labels = pc.getFiles(args.bi, args.suffix,
                                           labelfile=args.bl, concat=True)
#        bg_files = []
#        bg_labels = []
#        for e,bi in enumerate(args.bi):
#            tmp_bg_files, tmp_bg_labels = pc.getFiles(bi, args.suffix,
#                                      labelfile=args.bl[e])
# Don't need this assert since the background labels are allowed
# to appear multiple times
#            assert( len(list(set(tmp_bg_labels))) == len(tmp_bg_labels) )
#            bg_files.extend(tmp_bg_files)
#            bg_labels.extend(tmp_bg_labels)

#        assert( len(list(set(bg_labels+labels))) == len(bg_labels+labels) )
        assert( len(set(labels).intersection(set(bg_labels))) == 0)

    ex_cls = []
    if args.load_ex_cls:
        for f in files:
            ex_cls.append(pc.load(f))
    else:
        if (not args.scale and not args.load_trafo == 'scaler') and\
           ('svm' in args.clsname or args.clsname == 'sgd'):
            print 'WARNING: svm or sgd chosen but not --scale!' 
        
        all_cls = args.func(args)
        if not all_cls:
            raise ValueError('no classifier given')
        the_cls = all_cls[0]
      
        print 'load:', args.inputfolder
        descr = pc.loadDescriptors(files)
        print 'shape:', descr.shape
        if len(args.bi) > 0:
            print 'load descriptors of: ' + ','.join(args.bi)
            descr_bg = pc.loadDescriptors(bg_files)
            print 'shape:', descr_bg.shape
            if not args.attribute:
                descr = np.concatenate([descr, descr_bg], axis=0)
                print 'concat shape:', descr.shape

        print 'pre descr[0]', descr[0]
        print 'fit-transform'
        descr = prep.fit_transform(descr)
        print 'post descr[0]', descr[0]
        print 'possible new shape:', descr.shape
        prep.save_trafos(args.outputfolder)
            
        if args.attribute:
            descr_bg = prep.transform(descr_bg)
            print 'compute attribute space, dim=', len(descr_bg)
            ex_cls_bg = computeExCls(descr_bg, the_cls, len(descr_bg),
                                     args.outputfolder, bg_labels,
                                     '_attr.pkl.gz', parallel=args.parallel)
            descr = exemplar_cls.predictExemplarCls(descr, ex_cls_bg)
            # platt calibration 
#            ab_list = computeAB(descr_bg, ex_cls_bg, bg_labels)
#            descr = convertToProbs(descr, ab_list)
            print 'new descr-shape:', descr.shape

        ex_cls = computeExCls(descr, the_cls, len(files),
                              args.outputfolder, labels, parallel=args.parallel)
        
        # platt calibration 
#        ab_list = computeAB(descr, ex_cls, labels)
       
    print 'load test:', args.pi
    files_probe, labels_probe = pc.getFiles(args.pi, 
                                            args.suffix,
                                            labelfile=args.pl)

    print 'predict now'
    scores = predict(files_probe, ex_cls, prep, parallel=args.parallel)
    # this is our scores-matrix
    scores_mat = np.concatenate(scores, axis=0)
    stats = evaluate.computeStats('sum/max', scores_mat, 
                                  labels_probe, labels,
                                  distance=False, parallel=args.parallel)
    if write_stats: 
        evaluate.write_stats(os.path.join(args.outputfolder, 'stats.txt'), stats)


if __name__ == '__main__':
    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
               progressbar.ETA()]
    prep = preprocess.Preprocess.fromArgs()    
    parser = argparse.ArgumentParser(description="Exemplar Cls options",
                                     parents=[prep.parser])
    parser = pc.commonArguments(parser)
    parser = classification.parseArguments(parser)
    parser = parserArguments(parser)
    args = parser.parse_args()

    if args.log:
        log = pc.Log(sys.argv, args.outputfolder)

    if not args.outputfolder:
        raise ValueError('no outputfolder given')

    if not os.path.exists(args.outputfolder):
        pc.mkdir_p(args.outputfolder)

    run(args, prep, True)

    if args.log:
        log.dump()
