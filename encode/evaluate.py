import os
import numpy as np
import cv2
import sys
import argparse
import matplotlib.pyplot as plt
import gzip
import cPickle
import progressbar
import itertools
import glob

import scipy.spatial.distance as spdistance
from scipy import cluster

from sklearn.metrics import confusion_matrix, classification_report,\
                                       recall_score, roc_auc_score, f1_score,\
                                        precision_recall_fscore_support,\
                                        roc_curve, accuracy_score

from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation
from sklearn import decomposition
from sklearn.datasets import dump_svmlight_file
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report,\
                           recall_score, roc_auc_score, f1_score,\
                            precision_recall_fscore_support,\
                            roc_curve

import puhma_common as pc
import preprocess


# how books do it (and scikit learn)
def chiSquare(h1, h2):
	tmp = h1 - h2
	return np.sum( (2*tmp*tmp) / (h1 + h2 + np.finfo(float).eps ) )

# how opencv does it? (seems however worse than OpenCV)	
def chiSquare2(h1, h2):
	tmp = h1 - h2
	return np.sum( (tmp*tmp)/(h1+np.finfo(float).eps) )

# how the anonymous internet says it	
# Note: guess this one is actually wrong
def chiSquare3(h1, h2):
	tmp = h1 - h2
	return np.sum((tmp*tmp)/(h1+h2+np.finfo(float).eps))*0.5

def computeDistance(X, Y, method):
    if 'cosine' in method:
        dist = spdistance.cosine(X,Y)
    elif 'dot' in method:
        dist = 1.0 - X.dot(Y)
    elif 'chi2' in method:
        dist = chiSquare2(X, Y)
    elif 'chi3' in method:
        dist = chiSquare3(X, Y)
    elif 'chi' in method:
        dist = chiSquare(X, Y)
    elif 'euclidean' in method:
        dist = cv2.norm(X, Y)
    elif 'canberra' in method:
        dist = spdistance.canberra(X,Y)
    elif 'correl' in method:
        dist = spdistance.correlation(X,Y)
    else:
        # does that work?
        dist = cv2.compareHist(X, Y, method) 
   
    if hasattr(cv2, 'cv') and 'cv2.cv.CV_COMP_CORREL' in method:
        dist = 1 - dist
    elif hasattr(cv2, 'HISTCMP_CORREL') and 'cv2.HISTCMP_CORREL' in method :
        dist = 1 - dist
    elif hasattr(cv2, 'cv') and 'cv2.cv.CV_COMP_INTERSECT' in method:
        dist = 1 - dist
    elif hasattr(cv2, 'HISTCMP_INTERSECT') and 'cv2.HISTCMP_INTERSECT' in method:
        dist = 1 - dist

    return dist

def computeDistances(descriptors, method, distance=True, parallel=True,
                     distance_func=None, nprocs=4):
    num_desc = len(descriptors)

    if np.isnan(descriptors).any():
        raise ValueError('nan in descr!')
    if np.isinf(descriptors).any():
        raise ValueError('inf in descr!')

    for i in range(len(descriptors)):
        if not descriptors[i].any(): # faster
            print 'WARNING: complete row {} is 0'.format(i)

    indices = [(y,x) for y in range(num_desc-1) for x in range(y+1, num_desc)]
    def loop(ind): 
        if distance_func == None:
            try:
                dist = computeDistance(descriptors[ ind[0] ],descriptors[ ind[1] ], method)
            except:
                print 'method {} failed'.format(method)
                raise
        else: 
            dist = distance_func( descriptors[ ind[0] ],descriptors[ ind[1] ] )
        return dist

    if parallel:
        dists = pc.parmap(loop, indices, nprocs=nprocs)
    else:
        dists = map(loop, indices) 
  
    dense_vector = np.array( dists, dtype=float )

    if spdistance.is_valid_y(dense_vector, warning=True):
        dist_matrix = spdistance.squareform( dense_vector )
    else:
        print 'ERROR: not a valid condensed distance matrix!'
        n = dense_vector.shape[0]
        d = int(np.ceil(np.sqrt(n * 2)))
        should = d * (d - 1) / 2
        print '{} != {}, num: {}'.format(should, n, num_desc)
        sys.exit(1)
                    
    # do some checks
    if np.isnan(dist_matrix).any():
        print 'WARNING have a nan in the dist-matrix'
    if np.isinf(dist_matrix).any():
        print 'WARNING have a inf in the dist-matrix'
    
    if distance:
        if np.count_nonzero(dist_matrix == np.finfo(dist_matrix.dtype).max) > 0:
            raise ValueError('there is already a float-maximum')
        np.fill_diagonal(dist_matrix, np.finfo(dist_matrix.dtype).max)        
    else:
        if np.count_nonzero(dist_matrix == np.finfo(dist_matrix.dtype).min) > 0:
            raise ValueError('there is already a float-min')
        np.fill_diagonal(dist_matrix, np.finfo(dist_matrix.dtype).min)

    return dist_matrix #, dist_m

def computeDistances2(descr_probe, descr_gallery, method, parallel=True,
                     distance_func=None, nprocs=4):
    if np.isnan(descr_probe).any():
        raise ValueError('nan in descr_probe!')
    if np.isinf(descr_probe).any():
        raise ValueError('inf in descr_probe!')
    if np.isnan(descr_gallery).any():
        raise ValueError('nan in descr_galler!')
    if np.isinf(descr_gallery).any():
        raise ValueError('inf in descr_galler!')


    n_probes = len(descr_probe)
    n_gallery = len(descr_gallery)
    indices = [(y,x) for y in range(n_probes) for x in range(n_gallery)]
    def loop(ind): 
        if distance_func == None:
            try:
                dist = computeDistance(descr_probe[ ind[0] ],
                                       descr_gallery[ ind[1] ], method)
            except:
                print 'method {} failed'.format(method)
                raise
        else: 
            dist = distance_func( descriptors[ ind[0] ],descriptors[ ind[1] ] )

        return dist

    if parallel:
        dists = pc.parmap(loop, indices, nprocs=nprocs)
    else:
        dists = map(loop, indices)
  
    dense_vector = np.array( dists ).reshape(n_probes,-1)
    # do some checks
    if np.isnan(dense_vector).any():
        print 'WARNING have a nan in the dist-matrix'
    if np.isinf(dense_vector).any():
        print 'WARNING have a inf in the dist-matrix'

    return dense_vector

def computeStats( name, dist_matrix, labels_probe, 
                  labels_gallery=None, 
                  parallel=True, distance=True, nprocs=4,
                eval_method='cosine'):    
    n_probe, n_gallery = dist_matrix.shape
    # often enough we make a leave-one-out-cross-validation
    # here we don't have a separation probe / gallery
    if labels_gallery is None:
        n_gallery -= 1
        labels_gallery = labels_probe
        # assert not needed or?
        assert(dist_matrix.shape[0] == dist_matrix.shape[1])

    assert(dist_matrix.shape[0] == len(labels_probe))
    assert(dist_matrix.shape[1] == len(labels_gallery))

    # TODO: make variables choosable
    # Tolias et al. 2014 / 2016
    if 'poly' in eval_method:
        alpha = 3
        tau = 0
        sign = np.sign(dist_matrix)
        abso = np.abs(dist_matrix)
        abso = np.pow(abso[dist_matrix > tau], alpha)
        dist_matrix = sign * abso
    # Tao et al. 2014
    elif 'expo' in eval_method:
        beta = 10
        dist_matrix = np.exp(beta * dist_matrix)


    ind_probe = len(set(labels_probe))
    ind_gall = len( set(labels_gallery))

    labels_gallery = np.array(labels_gallery)
    labels_probe = np.array(labels_probe)

    print 'number of probes: {}, individuals: {}'.format(n_probe, ind_probe)
    print 'number of gallery: {}, individuals: {}'.format(n_gallery, ind_probe)
    if parallel:
        def sortdist(split):
            return split.argsort()
        splits = np.array_split(dist_matrix, 8) # todo assume 8 threads
        indices = pc.parmap(sortdist, splits, nprocs=nprocs)
        indices = np.concatenate(indices, axis=0)
    else:
        indices = dist_matrix.argsort()

    if not distance:
        indices = indices[:,::-1]

    def loop_descr(r):        
        rel_list = np.zeros((1,n_gallery))
        not_correct = []
        for k in range(0, n_gallery): 
            if labels_gallery[ indices[r,k] ] == labels_probe[ r ]:
                rel_list[0,k] = 1
            elif k == 1:
                not_correct.append((r,indices[r,k]))
        return rel_list, not_correct

    if parallel:
        all_rel, top1_fail = zip(*pc.parmap(loop_descr, range(n_probe), nprocs=nprocs ))
    else:
        all_rel, top1_fail = zip(*map(loop_descr, range(n_probe)) )

    # make all computations with the rel-matrix
    rel_conc = np.concatenate(all_rel, 0)
    # are there any zero rows?
    z_rows = np.sum(rel_conc,1)
    n_real2 = np.count_nonzero(z_rows)
    if n_real2 != rel_conc.shape[0]:
        print('WARNING: not for each query exist also a label in the gallery'
                '({} / {})'.format(n_real2, len(rel_conc.shape[0])))
    rel_mat = rel_conc[z_rows > 0]
    print 'rel_mat.shape:', rel_mat.shape
    prec_mat = np.zeros(rel_mat.shape)
    
    soft2 = np.zeros(50)
    hard2 = np.zeros(4)
    for i in range(n_gallery):
        rel_sum = np.sum(rel_mat[:,:i+1], 1)
        prec_mat[:,i] = rel_sum / (i+1)
        if i < 50:
            soft2[i] = np.count_nonzero(rel_sum > 0) / float(n_real2)
        if i < 4:
            hh = rel_sum[np.isclose(rel_sum,(i+1))]
#            print 'i: {} len(hh): {}'.format(i, len(hh))
            hard2[i] = len(hh) / float(n_real2)

    map2 = np.mean(prec_mat[rel_mat==1])

    print 'correct: {} / {}'.format(np.sum(rel_mat[:,0]), n_real2)
    print 'map:', map2
    print 'top-k soft:', soft2[:10]
    print 'top-k hard:', hard2
      
    # Average precisions
    ap = []
    for i in range(n_real2):
        ap.append(np.mean(prec_mat[i][rel_mat[i]==1]))

    print 'mean(ap):',np.mean(ap)
    print 'isclose(map2, mean(ap)): {}'.format(np.isclose(map2,np.mean(ap)))

    # precision@x scores
    p2 = np.sum(prec_mat[:,1]) / n_real2
    p3 = np.sum(prec_mat[:,2]) / n_real2
    p4 = np.sum(prec_mat[:,3]) / n_real2
    print 'mean P@2,P@3,P@4:',p2,p3,p4
    stats = {'topx_soft':soft2[:10],
             'topx_hard':hard2,
             'mAP':map2,
             'top1_fail':top1_fail,
             'ap': ap,
             'p2':p2, 'p3':p3, 'p4':p4}
    return stats


# TODO: Revise: make a class out of this! 
# giving parameters all the time sucks...
def runNN(descriptors, labels, distance=True, histogram=False, 
          parallel=True, eval_method='cosine', nprocs=None):
    """
    compute nearest neighbor from specific descriptors, given labels
    """

    distance_method = {
                       "dot":'dot',\
                       "euclidean": 'euclidean',\
                       "cosine": 'cosine',
                       "canberra": 'canberra',\
                       "cosinem": 'cosine',\
                       "pca": 'cosine',\
                       "pca_full": 'cosine',\
                       "cosinef": 'cosine'
                      }
    if histogram:
        distance_method.update({
                       "chi": 'chi', \
                       "hellinger": cv2.cv.CV_COMP_HELLINGER,\
                       "intersect": cv2.cv.CV_COMP_INTERSECT
                        })

    name = eval_method
    ret_matrix = None
    method = distance_method[name]
    dist_matrix2 = None
    if 'cosinem' in name:
        alpha = 0.8
        fac = alpha*np.mean(descriptors, axis=0)
        dist_matrix = computeDistances(descriptors - fac, method, distance,
                                      parallel, nprocs=nprocs)
    elif 'cosinef' in name:
        alpha = 0.8
        dist_matrix = computeDistances(np.sign(descriptors)*np.power(np.abs(descriptors), alpha), method,
                                      distance, parallel, nprocs=nprocs)
    elif 'pca' in name:            
        pca = decomposition.PCA(min(128,descriptors.shape[1]), whiten=False)
        pca_descr = pca.fit_transform(descriptors)
        dist_matrix = computeDistances(pca_descr, method, distance, parallel,
                                       nprocs=nprocs)
    elif 'pca_full' in name:
        pca = decomposition.PCA(descriptors.shape[1], whiten=False)
        pca_descr = pca.fit_transform(descriptors)
        dist_matrix = computeDistances(pca_descr, method, distance, parallel,
                                       nprocs=nprocs)
    else:
        dist_matrix = computeDistances(descriptors, method, distance,
                                       parallel, nprocs=nprocs)

    if labels is not None:
        stats = computeStats(name, dist_matrix, labels, 
                             parallel=parallel, distance=distance, nprocs=nprocs,
                             eval_method=eval_method)
    else:
        print 'WARNING: labels is None -> no stats'
        stats = None

    ret_matrix = dist_matrix # is there a case where that is not the case??
    return ret_matrix, stats

def writeLabels(ret_matrix, gallery_labels, probe_files,
                outputfolder, suffix,
                distance=True,                
                fname='labels_out.txt'):
    if np.isnan(ret_matrix).any():
        print 'WARNING have a nan in the ret_matrix, writeLabels()'
    if np.isinf(ret_matrix).any():
        print 'WARNING have a inf in the ret_matrix, writeLabels()'

    if len(probe_files) != ret_matrix.shape[0]:
        raise ValueError( '{} != {}'.format(len(probe_files),
                                            ret_matrix.shape[0]))
   
    ind = ret_matrix.argsort()
    if not distance:
        ind = ind[:,::-1]

    out_labels = []
    for i in range(ret_matrix.shape[0]):
        out_labels.append( gallery_labels[ ind[i][0] ] )
       
    full_name = os.path.join(outputfolder, fname)
    with open(full_name, 'w') as f:
        for i,pf in enumerate(probe_files):
            to_f = os.path.basename(pf)
            if suffix == '':
                to_f = os.path.splitext(to_f)[0]
            else: 
                to_f = to_f.replace(suffix, '')
            f.write('{} {}\n'.format(to_f, out_labels[i]))
    print 'wrote', full_name

def classifyNN(descr_probe, descr, labels_probe, labels,
               eval_method='cosine',
               distance=True, parallel=False, nprocs=None,
              ret_matrix=None):
    if ret_matrix is None:
        ret_matrix = computeDistances2(descr_probe, descr, eval_method,
                                       parallel=parallel, nprocs=nprocs) 
        print 'ret-matrix.shape:', ret_matrix.shape
    stats_tmp = computeStats('cosine', ret_matrix, labels_probe,
                             labels, parallel=parallel,
                             distance=distance, nprocs=nprocs,
                            eval_method=eval_method)

    indices = ret_matrix.argsort()
    if not distance:
        indices = indices[:,::-1]

    le = preprocessing.LabelEncoder()
    le.fit(labels)
    try:
        t_labels = le.transform(labels_probe)
    except ValueError:
        print 'WARNING: labels_probe not in labels -> skip further evaluation'
    else:
    #    print len(labels_probe), indices.shape
        pred_labels = le.transform([ labels[indices[i,0]] for i in\
                                    range(len(labels_probe))])
        p, r, f, _ = precision_recall_fscore_support(pred_labels,
                                             t_labels,
                                             average='micro')
        print 'p, r, f1:', p, r, f
        stats_tmp['f1'] = f
        stats_tmp['pr'] = p
        stats_tmp['re'] = r
    #    cm = confusion_matrix(pred_labels, t_labels)
    #    print cm
    #    print le.inverse_transform(np.arange(len(le.classes_)))
        print (classification_report(pred_labels, t_labels))
        acc = accuracy_score(pred_labels, t_labels) 
        print 'accuracy:', acc
        stats_tmp['acc'] = acc

        # kinda cheating: if we know that there exist only one instance
        out_labels = [ 'lalelu' for i in range(len(labels_probe)) ]
        for c in range(len(labels_probe)):
            # find best one
            if not distance:
                max_ind = ret_matrix.argmax()
            else: 
                max_ind = ret_matrix.argmin()
            # probe and gallery index
            p_ind, g_ind = np.unravel_index(max_ind, ret_matrix.shape)
            # set corresponding label for that probe 
            out_labels[p_ind] = labels[ g_ind ]
            # 'remove' row and column from scores
            if not distance:
                ret_matrix[ p_ind, : ] = np.finfo(ret_matrix.dtype).min
                ret_matrix[ :, g_ind ] = np.finfo(ret_matrix.dtype).min
            else:
                ret_matrix[ p_ind, : ] = np.finfo(ret_matrix.dtype).max
                ret_matrix[ :, g_ind ] = np.finfo(ret_matrix.dtype).max

        corr = np.count_nonzero(np.array(labels_probe) == np.array(out_labels))
        print 'New TEST w.o. doubles: top1= {}'.format(float(corr) / float(len(labels_probe)))

    return ret_matrix, stats_tmp

def write_stats(stats_fname, stats, identifier):
    def remove_empty(a):
        ret = []
        for el in a:
            if el != ():
                ret.append(el)
        return ret

    if not stats_fname.endswith('txt'):
        base = stats_fname
        stats_fname += '.txt'
    else:
        base = os.path.splitext(stats_fname)[0]

    with open(stats_fname, 'a') as f:
        if identifier and identifier != '':
            f.write('{}\n'.format(identifier))
        if 'mAP' in stats:
            f.write('mAP\n{}\n'.format(stats['mAP']))
        if 'topx_soft' in stats:
            f.write('topx_soft\n' + ' '.join(map(str,stats['topx_soft'])) + '\n')
        if 'topx_hard' in stats:
            f.write('topx_hard\n' + ' '.join(map(str,stats['topx_hard'])) + '\n')
        # all single stats:
        for k in ['mAP', 'acc', 'f1', 'pr', 're', 'r2', 'rmse', 'mae',
                  'mae-std', 'medae', 'spear', 'spear-p', 'p2', 'p3',
                  'p4','kappa', 'rmac', 'rmic', 'pmic', 'pmac', 'fmic', 'fmac']:
            if k in stats:
                f.write('{}\n{}\n'.format(k,stats[k]))
        if 'cs' in stats:
            f.write('cs\n' + ' '.join(map(str,stats['cs'])) + '\n')

    print 'wrote', stats_fname

    # single
    for k in ['mAP', 'cs', 'acc', 'rmic', 'rmac', 'pmic','pmac','fmac','fmic',
              'kappa']:
        if k in stats:
            if k == 'kappa':
                with open(base + '_{}.txt'.format(k.lower()), 'a') as f:
                    f.write('{:.4f}\n'.format(stats[k]))
            else:
                with open(base + '_{}.txt'.format(k.lower()), 'a') as f:
                    f.write('{:2.2f}\n'.format(100*stats[k]))

        
    if 'topx_soft' in stats:
        with open(base + '_soft.txt', 'a') as f:
            f.write(' '.join(map(str,stats['topx_soft'])) + '\n')
    if 'topx_hard' in stats:
        with open(base + '_hard.txt', 'a') as f:
            f.write(' '.join(map(str,stats['topx_hard'])) + '\n')
    
    if 'top1_fail' in stats:
        with open(base + '_top1_fail.txt', 'a') as f:
            f.write('top1_fail\n' + ' '.join(map(str,remove_empty(stats['top1_fail']))) + '\n')

    if 'rmac' in stats and 'rmic' in stats and 'kappa' in stats:
        with open(os.path.splitext(stats_fname)[0] + '_amir.tex', 'a') as f:
            f.write('{} & {:2.2f} & {:2.2f} & {:.4f}\n'.format(identifier, 
                                                           100*stats['rmac'], 
                                                           100*stats['rmic'], 
                                                           stats['kappa']))
    # tex-table
    def writeTex(suffix='', map_pos_middle=True, h4=True, p4=True):
        header = False
        if not os.path.exists(base + suffix + '.tex'):
            header = True
            
        metric_list = ['acc', 'f1', 'pr', 're', 
                                   'p2', 'p3']
        regression_list = ['r2', 'rmse', 'mae', 'mae-std', 'medae',
                                       'spear', 'spear-p']
        if p4:
            metric_list.append('p4')
        if map_pos_middle:
            metric_list.insert(0,'mAP')
        else:
            metric_list.append('mAP')

        full_metric_list = metric_list + regression_list

        with open(base + suffix + '.tex', 'a') as f:
            # create the header
            if header:
                if 'topx_hard' in stats:
                    f.write(' & Top-1 & Hard-2 & Hard-3')
                    if h4: f.write(' & Hard-4')
                if 'topx_soft' in stats:
                    f.write(' & Soft-5 & Soft-10')
               
                for k in full_metric_list:
                    if k in stats:
                        f.write('& {}'.format(k))
                if 'cs' in stats:
                    f.write(' & cs25 & cs50')
                f.write('\n')

            f.write('{}'.format(identifier))
            if 'topx_hard' in stats:
                if h4:
                    f.write(' & ' + \
                        ' & '.join(map('{:.2f}'.format,
                                       100*np.array(stats['topx_hard']))))
                else:
                    f.write(' & ' + \
                        ' & '.join(map('{:.2f}'.format,
                                       100*np.array(stats['topx_hard'][:3]))))
            if 'topx_soft' in stats:
                f.write(' & {:.2f} & {:.2f}'.format(100*stats['topx_soft'][4],
                                                      100*stats['topx_soft'][9]))
                
            for k in metric_list:
                if k in stats:
                    f.write(' & {:.2f}'.format(100*stats[k]))
            # regression stuff 
            for k in regression_list:
                if k in stats:
                    f.write(' & {:.2f}'.format(stats[k]))
            if 'cs' in stats:
                f.write(' & {:.2f} & {:.2f}'.format(100*stats['cs'][4],
                                                      100*stats['cs'][9]))

            f.write('\\\\\n')

    writeTex()
    writeTex('2', False, False, False)

def parserArguments(parser):
    parser.add_argument('--dist_matrix', 
                        help='distance-matrix')
    parser.add_argument('--dist_matrix2', 
                       help='second distance matrix for p-test')
    parser.add_argument('--fusion', 
                        default='early', 
                        choices=['early', 'earlycombi', 'mid', 'late', 'none'],
                        help='fusion technique')
    parser.add_argument('--cluster',
                        help='cluster the results and print cluster results and'
                        ' dendogram')
    parser.add_argument('--stats_filename', default='stats.txt',
                        help='filename for the stats file')
    parser.add_argument('--labelfile_probe', 
                        help='use extra query labels, the other labels target'
                        ' is for the gallery')
    parser.add_argument('--inputfolders_probe', 
                        nargs='+',
                        help='give some inputfolders or multiple ones')
    parser.add_argument('--inputfolders_suffix_probe',
                        default='',
                        help='if a suffix should be appended after'
                        ' inputfolder/*/inputfolders_suffix')
    parser.add_argument('--write_labels', action='store_true',
                        help='write predicted labels out')
    parser.add_argument('--eval_method', default='cosine',
                        help='evaluation method')

    return parser

def run(args, prep=None, identifier=''):
    if prep == None:
        prep = preprocess.Preprocess()
    if args.dist_matrix:       
        files, labels = pc.getFiles(args.inputfolder, args.suffix, labelfile=args.labelfile, 
                                    exact=True)
        dist_matrix = np.loadtxt(args.dist_matrix, delimiter=',',ndmin=2,
                                 dtype=np.float64)
        stats_d1 = computeStats( 'cosine', dist_matrix, labels,
                                parallel=args.parallel, distance=True,
                                nprocs=args.nprocs,
                               eval_method=args.eval_method)
        if args.outputfolder:
            write_stats(os.path.join(args.outputfolder, args.stats_filename),
                        stats_d1, args.identifier)
        if args.dist_matrix2:
            dist_matrix = np.loadtxt(args.dist_matrix2, delimiter=',',ndmin=2,
                                 dtype=np.float64)
            stats_d2 = computeStats( 'cosine', dist_matrix, labels,
                                    parallel=args.parallel, distance=True,
                                    nprocs=args.nprocs,
                                   eval_method=args.eval_method)
            # make p-test
            from scipy import stats
            s1 = stats_d1['ap']
            s2 = stats_d2['ap']
            T, p = stats.wilcoxon(s1,s2)
            print 'wilcox T:', T
            print 'wilcox p:', p
            
            k, p = stats.normaltest(s1)
            print 'normaltest1 k:', k
            print 'normaltest1 p:', p
            k, p = stats.normaltest(s2)
            print 'normaltest2 k:', k
            print 'normaltest2 p:', p

            f,p = stats.f_oneway(s1,s2)
            print 'anova:', f, p

            h, p = stats.kruskal(s1,s2)
            print 'kruskal h, p', h, p

            print 'pearson:', stats.pearsonr(s1,s2)

            t, p = stats.ttest_ind(s1,s2)
            print 't-test:', t, p
            
            t, p = stats.ttest_ind(s1,s2,equal_var=False)
            print 't-test (false):', t, p

            def exact_mc_perm_test(xs, ys, nmc):
                n, k = len(xs), 0.0
                diff = np.abs(np.mean(xs) - np.mean(ys))
                zs = np.concatenate([xs, ys])
                for j in range(nmc):
                    np.random.shuffle(zs)
                    k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
                return k / float(nmc)
            
            print 'try exact monte carlo permutation test'
            p = exact_mc_perm_test(stats_d1['ap'], stats_d2['ap'], 10000)
            print 'exact mc:', p

            def permutation_resampling(case, control, num_samples, statistic):
                """Returns p-value that statistic for case is different
                from statistc for control."""

                observed_diff = abs(statistic(case) - statistic(control))
                num_case = len(case)

                combined = np.concatenate([case, control])
                diffs = []
                for i in range(num_samples):
                    xs = np.random.permutation(combined)
                    diff = np.mean(xs[:num_case]) - np.mean(xs[num_case:])
                    diffs.append(diff)

                pval = (np.sum(diffs > observed_diff) +
                        np.sum(diffs < -observed_diff))/float(num_samples)
                #return pval, observed_diff, diffs
                return pval
            
            print 'permutation test', permutation_resampling(s1,s2,10000,np.mean)


        sys.exit(0)

    
    files, labels = pc.getFiles(args.inputfolder, args.suffix,                                 
                                labelfile=args.labelfile, 
                                inputfolders_suffix=args.inputfolders_suffix)
    if args.fusion == 'early':
        descriptors = [ pc.loadDescriptors(files) ]
        print 'loaded descriptor(s), shape:', descriptors[0].shape
    else:
        raise ValueError('currently no other fusion than <early> allowed!')

    # concatenate all possible features
#    if len(args.inputfolder) > 1 or args.inputfolders_suffix != '':
#
#        descriptors, labels, all_files = pc.loadAllDescriptors(args.inputfolder,
#                                                    args.inputfolders_suffix,
#                                                    args.suffix, args.labelfile,
#                                                    1 if args.fusion == 'early' else None)
    # TODO: this is unlogic: should be args.labelfile_gallery ...
    if args.labelfile_probe:
        if args.inputfolders_probe:
            probe_inputfolders = args.inputfolders_probe
        else:
            probe_inputfolders = args.inputfolder
        if args.inputfolders_suffix_probe:
            probe_inputfolders_suffix = args.inputfolders_suffix_probe
        else:
            probe_inputfolders_suffix = args.inputfolders_suffix

        descriptors_probe, labels_probe, files_probe =\
                pc.loadAllDescriptors(probe_inputfolders, 
                                      probe_inputfolders_suffix,
                                      args.suffix,
                                      args.labelfile_probe,
                                      1 if args.fusion == 'early' else None)

    stats = []
    if 'early' == args.fusion:
        if 'fit' in args.mode:
            prep.fit(descriptors[0])
            prep.save_trafos(args.outputfolder)
            descr = descriptors[0]
        elif 'transform' in args.mode:
            descr = prep.transform(descriptors[0])
            if args.labelfile_probe:
                descr_probe = prep.transform(descriptors_probe[0])
                print 'descr_probe: new shape:', descr_probe.shape
# doesn't seem to be needed                
#                descr = preprocessing.normalize(descr, norm='l2', copy=False)
            print 'new shape:', descr.shape
        else:
            descr = descriptors[0]
            if args.labelfile_probe:
                descr_probe = descriptors_probe[0]
        
        if args.labelfile_probe:              
            ret_matrix, stats_tmp = classifyNN(descr_probe, descr, 
                                               labels_probe, labels, 
                                               distance=True,
                                               eval_method=args.eval_method,
                                               parallel=args.parallel, 
                                               nprocs=args.nprocs)

        else:
            ret_matrix, stats_tmp = runNN( descr, labels, distance=True,
                                          histogram=False,
                                          eval_method=args.eval_method,
                                          parallel=args.parallel, 
                                          nprocs=args.nprocs)
        stats.append(stats_tmp)

    if 'earlycombi' == args.fusion:
        # powerset
        def powerset(iterable):
            "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
            s = list(iterable)
            return \
                list(itertools.chain.from_iterable(itertools.combinations(s, r) for\
                                           r in range(len(s)+1)))
        tmp_infolds = powerset(inputfolders)            
        tmp_descr = powerset(descriptors)
        infolds = []
        comb_descr = []
        for i in range(len(tmp_descr)):                
            if len(tmp_descr[i]) == 0: continue
            elif len(tmp_descr[i]) == 1: 
                comb_descr.append(tmp_descr[i][0])
                infolds.append(tmp_infolds[i][0])
            else:
                comb_descr.append(np.concatenate(list(tmp_descr[i]),
                                                  axis=1))
                infolds.append(' + '.join(tmp_infolds[i]))
        descriptors = comb_descr

#        assert( isinstance(descriptors, list) )
#        rets = []
#        for i, descr in enumerate(descriptors):
#            if 'earlycombi' == args.fusion:
#                print infolds[i]
#            if args.labelfile_probe:               
#                ret_matrix = computeDistances2(descr_probe, descr, 'cosine') 
#                stats_tmp = computeStats2('cosine', ret_matrix, labels_probe, labels)
#            else:
#                ret_matrix, stats_tmp = runNN( descr, labels, distance=True,
#                                   histogram=False)
#
#            stats.append(stats_tmp)
#            rets.append(ret_matrix)
#    
#        descr = descriptors
       

    if args.fusion == 'mid' and len(rets) == 2:
        weights = [ 0.1 * i for i in range(1,10)]
        for w1 in weights:
            descr[0] *= w1
            for w2 in weights:
                print 'w1: {}, w2: {}'.format(w1,w2)
#                    descr[1] *= w2
                descr[0][:,:descr[1].shape[1]] += w2 * descr[1]
                ret_matrix, stats_tmp = runNN(descr[0], labels, distance=True,
                                              histogram=False, nprocs=args.nprocs,
                                              eval_method=args.eval_method)
                stats += stats_tmp
#                    files, _ = pc.getFiles(args.inputfolders[0], args.patterns[0], labelfile=args.labelfile, 
#                                      exact=True)
#                    descr[0] = pc.loadDescriptors(files)
                
            files, stats_tmp = pc.getFiles(inputfolder[0], args.suffix, labelfile=args.labelfile, 
                                  exact=True)
            stats.append(stats_tmp)
            descr[0] = pc.loadDescriptors(files)


    elif args.fusion == 'late':
        # TODO: relaxe this late fusion
        if len(rets) == 2:
            weights = [ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for w in weights:
                print w
                mat = rets[0] * w + rets[1] * (1-w)
                stats_tmp = computeStats('combined', mat, labels,
                                         labels_gallery=None,
                                         parallel=args.parallel,
                                         distance=True, 
                                        nprocs=args.nprocs,
                                        eval_method=args.eval_method) 
                stats.append(stats_tmp)

    if args.outputfolder:
        fname, fname_ext = os.path.splitext(args.stats_filename)
        for i,s in enumerate(stats):
            write_stats(os.path.join(args.outputfolder, 
                                     fname + '_' + str(i) + fname_ext), 
                    s, args.identifier)
        fpath = os.path.join(args.outputfolder, 'dist.cvs')
        np.savetxt(fpath, ret_matrix, delimiter=',')
        print 'saved', fpath

    if args.write_labels:
        writeLabels(ret_matrix, labels,
                    files_probe[0] if args.labelfile_probe else files,
                    args.outputfolder, args.suffix)

if __name__ == '__main__':
#    import stacktracer
#    stacktracer.trace_start("trace.html",interval=5,auto=True)
    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
               progressbar.ETA()]
    prep = preprocess.Preprocess.fromArgs()    
    parser = argparse.ArgumentParser(description="Evaluate stuff",
                                     parents=[prep.parser])
    parser = pc.commonArguments(parser)
    parser = parserArguments(parser)

    args = parser.parse_args()
        
    if args.log: 
        log = pc.Log(sys.argv, '.' if not args.outputfolder else\
                         args.outputfolder)

    if args.outputfolder and not os.path.exists(args.outputfolder):
        pc.mkdir_p(args.outputfolder)
    
    run(args, prep)

    if args.log:
        log.dump()
