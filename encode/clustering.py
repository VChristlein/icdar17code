import os
import glob
import gzip
import cPickle
import argparse
import numpy as np
import sys
from sklearn import cluster
from sklearn import decomposition
from sklearn import mixture
from sklearn import preprocessing
from scipy import stats as stats
import puhma_common as pc
import cv2
import progressbar
import preprocess

def fitGMM(data, num_clusters, iterations, update, covar_type, nprocs=1):
    print ('fit gmm with num_clusters {}, params {}, n_iter {}, covar_type '
        '{}'.format(num_clusters, update, iterations, covar_type))
    try:
        gmm = mixture.GaussianMixture(num_clusters, max_iter=iterations,
                                      covariance_type=covar_type, 
                                      n_init=1)
    except:
        print "WARNING Unexpected error:", sys.exc_info()[0]
        print 'continue with old sklearn GMM'
        gmm = mixture.GMM(num_clusters, n_iter=iterations,
                      params=update, covariance_type=covar_type, 
                      n_init=1)
#                     init_params='wc')
#    gmm.means_ = cluster.KMeans(n_clusters=num_clusters,
#                               max_iter=1, n_jobs=nprocs).fit(data).cluster_centers_    
    gmm.fit(data)
    if not hasattr(gmm, 'covars_'): # sklearn changed the variable name :/
        gmm.covars_ = gmm.covariances_
    return gmm


def computeVocabulary(descriptors, method, num_clusters, iterations, update,
                      lib, covar_type,  nprocs=1):
    print 'compute now vocabulary, method:', method
    if 'sparse' in method:
        dl = decomposition.DictionaryLearning(num_clusters, max_iter=iterations)
        dl.fit(descriptors)
        return np.array(dl.components_)

    elif 'vgmm' in method:
        if 'vgmm2' == method:
            gmm = mixture.BayesianGaussianMixture(num_clusters,
                                              covariance_type=covar_type,
                                              weight_concentration_prior_type='dirichlet_distribution')
        else:
            gmm = mixture.BayesianGaussianMixture(num_clusters,
                                              covariance_type=covar_type,
                                              weight_concentration_prior_type='dirichlet_process')
        gmm.fit(descriptors)
        trainer = gmm
        trainer.type_ = 'gmm'

    elif 'gmm' == method:
        if 'cv2' in lib:
            # FIXME add iterations parameter (standard: 100)
            try:
                em = cv2.ml.EM_create()
                em.setClustersNumber(num_clusters)
                em.trainEM(descriptors)
                means = em.getMeans()
                weights = em.getWeights()
                covs_ = em.getCovs()
            except e:
                print 'WARNING: got exception {}\ntry old EM'.format(e)
                em = cv2.EM(num_clusters, cv2.EM_COV_MAT_DIAGONAL)
                em.train(descriptors)
                means = em.getMat('means')
                weights = em.getMat('weights')
                covs_ = em.getMatVector('covs')
            
            # convert to sklearn gmm
            covs = np.array([np.diagonal(c) for c in covs_])
            print means.shape, weights.shape, len(covs_), covs.shape
            gmm = mixture.GMM(num_clusters)
            gmm.weights_ = weights.flatten()
            gmm.means_ = means
            gmm._set_covars( covs )
        else:
            gmm = fitGMM(descriptors, num_clusters,
                         iterations, update, covar_type, nprocs)
        trainer = gmm
        trainer.type_ = 'gmm'
    elif method == 'fast-gmm':
        means = cluster.MiniBatchKMeans(num_clusters,
                                        compute_labels=False,
                                        batch_size=100*num_clusters).fit(descriptors).cluster_centers_
        gmm = mixture.GaussianMixture(num_clusters, max_iter=1,
                                      covariance_type=covar_type, 
                                      n_init=1,
                                      means_init=means)
        gmm.fit(descriptors)
        trainer = gmm
        trainer.type_ = 'gmm'

    elif method == 'hier-kmeans':
        print 'run hierarchical kmeans'
        import pyflann
        flann = pyflann.FLANN(centers_init='kmeanspp')
        branch_size = 32
        num_branches = (num_clusters - 1) / (branch_size - 1)
        clusters = flann.hierarchical_kmeans(descriptors, branch_size, num_branches,
                                             iterations,
                                             centers_init='kmeanspp') 
        trainer = cluster.KMeans(num_clusters)
        trainer.cluster_centers_ = clusters
    elif method == 'kmeans':
        trainer = cluster.KMeans(num_clusters)
        if 'cv2' in lib:
            term_crit = (cv2.TERM_CRITERIA_EPS, 100, 0.01)
            ret, labels, clustes = cv2.kmeans(descriptors, num_clusters, term_crit, 10,\
                                              cv2.KMEANS_PP_CENTERS)
            trainer.cluster_centers_ = clusters
        else:
            trainer.fit(descriptors)
            #clusters = trainer.cluster_centers_.astype(np.float32)
    else:
        if method == 'mini-kmeans':
            trainer = cluster.MiniBatchKMeans(num_clusters,
                                              compute_labels=False,
                                              batch_size=10000 if num_clusters < 1000 else 50000)
        elif method == 'mean-shift':
            trainer = cluster.MeanShift()
        else:
            print 'unknown clustering method'
            sys.exit(1)
        trainer.fit(descriptors)
        #clusters = trainer.cluster_centers_.astype(np.float32)

    if not hasattr(trainer, 'means_'):
        trainer.means_ = trainer.cluster_centers_
        trainer.type_ = 'kmeans'

    return trainer

def addArguments(parser):	
    group = parser.add_argument_group('clustering options')
    group.add_argument('--num_clusters', type=int, default=100,\
                        help='number of cluster-centers = size of vocabulary')
    group.add_argument('--vocabulary_filename',\
                        help='write vocabulary to this file')
    group.add_argument('--method', default='gmm',\
                        choices=['mini-kmeans', 'mean-shift', 'hier-kmeans',
                                 'gmm', 'fast-gmm', 
                                 'sparse', 'sparsegmm', 'sparsemini',
                                 'sparseminigmm', 'kmeans', 'multi-kmeans',
                                 'multi-gmm',
                                 'mean-shift-direct', 'hierarchy', 'vgmm',
                                 'vgmm2', 
                                 'posteriors'], # means from posteriors
                        help=("method to use, standard is \'mini-kmeans\' for a"
                        "hierarchical kmeans, other option: \'mini-kmeans\', "
                        "\'mean-shift\', 'gmm' (Gaussian Mixture Model) "
                        " or 'sparse': sparse-coding, followed maybe by gmm "))
    group.add_argument('--linkage',
                       choices=['single', 'complete', 'average', 'weighted',
                                'centroid', 'median', 'ward'],
                       default='single',
                       help='linkage method for hierarchical clustering')
    group.add_argument('--iterations', type=int, default=100,\
                        help=' number of iterations (if gmm, this is the gmm '
                        'part, not the kmeans-initialization part)')
    group.add_argument('--gmm_update', default='wmc',\
                        help='what to update w. GMM, w:weights, m:means, c:covars')
    group.add_argument('--covar_type', default='diag',
                        choices=['full','diag'],
                        help='cocariance type for gmm')
    group.add_argument('--predict', default=False, action='store_true',
                        help='predict labels of the clusters (if supported)')
    return parser

def runStump(descriptors, args):
    return computeVocabulary(descriptors, args.method, args.num_clusters, 
                             args.iterations, args.gmm_update, args.lib,
                             args.covar_type, 
                             args.nprocs) 

def recomputeMeans(data, assignments, hard_assignment=True):
    """
    recompute mean for given data and assignment
    """
    if not hard_assignment:
        def compute(k):
            sum_ass = assignments[:,k].sum()
            if sum_ass != 0:
                return assignments[:,k].T.dot(data) / sum_ass
            else:
                return np.zeros(data.shape[1], data.dtype)
        means = map(compute, range( assignments.shape[1]))
        means_ = np.concatenate(means).reshape(assignments.shape[1], -1)

    else:
        if assignments.ndim == 1:
            n_clusters = len(set(assignments))
            means_ = np.zeros( (n_clusters, data.shape[1]),
                              dtype=data.dtype)
            for i in range(n_clusters):      
                for_mean = data[ assignments == i ]
                if len(for_mean) > 0:
                    means_[i] = np.mean( for_mean, axis=0 )

        else:
            # this is the same as above but much faster:
            means_ = np.zeros( (assignments.shape[1], data.shape[1]),
                              dtype=data.dtype)
            for i in range(assignments.shape[1] ):      
                for_mean = data[ assignments[:,i] > 0 ]
                if len(for_mean) > 0:
                    means_[i] = np.mean( for_mean, axis=0 )
    
    return means_

def run(args, prep=None):
    if prep == None:
        prep = preprocess.Preprocess()
    if not os.path.exists(args.outputfolder):
        pc.mkdir_p(args.outputfolder)

    files, labels = pc.getFiles(args.inputfolder, args.suffix, args.labelfile,
                           exact=args.exact,max_files=args.max_files)
    if files is None or len(files) == 0:
        print 'getFiles() returned no images'
        sys.exit(1)

    maskfiles = pc.getMaskFiles(files, args.suffix, args.maskfolder,
                                    args.masksuffix)
    if len(args.max_descriptors) == 0:
        descriptors, rand_indices = pc.loadDescriptors(files, rand=True, 
                                                 return_random_indices=True)
    else:
        max_descs_per_file = int(args.max_descriptors[0] / float(len(files)))
        max_descs_per_file = max(max_descs_per_file, 1)
        descriptors, rand_indices = pc.loadDescriptors(files,\
                                                        max_descs=args.max_descriptors[0],
                                                        max_descs_per_file=max_descs_per_file,
                                                        rand=True, 
                                                        maskfiles=maskfiles,
                                                        return_random_indices=True) 

    print 'got {} features'.format(len(descriptors))
    print 'features.shape', descriptors.shape
    
    # load features to train a universal background gmm
    print 'load features for training ubm from {} files'.format(len(files))
    
    if args.method == 'posteriors':
        posteriors_files, _ = pc.getFiles(args.posteriors_dir, 
                                       args.posteriors_suffix,
                                       labelfile=args.labelfile, 
                                       exact=args.exact,
                                       max_files=args.max_files)
        assert(len(posteriors_files) == len(files))
        indices = []

        widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
               progressbar.ETA()]
        progress = progressbar.ProgressBar(widgets=widgets,
                                           maxval=len(posteriors_files))
        progress.start()
        for e,f in enumerate(posteriors_files):
            posteriors = pc.loadDescriptors(f)
            posteriors = posteriors[ rand_indices[e] ]
            cluster_idx = posteriors.argmax(axis=1)
            indices.append(cluster_idx)
            progress.update(e+1)
        progress.finish()
        
        indices = np.concatenate(indices)
        assert(len(indices) == len(descriptors))
        means = recomputeMeans(descriptors, indices)
        vocabulary = cluster.KMeans(means.shape[0]) # dummy
        vocabulary.means_ = means
        vocabulary.type_ = 'kmeans'
    else:
        vocabulary = computeVocabulary(descriptors, args.method, args.num_clusters, 
                                   args.iterations, args.gmm_update, args.lib,
                                   args.covar_type,
                                   args.nprocs) 


    # TODO: rewrite to be more generic
    if 'sparse' in args.method and 'gmm' in args.method:
        gmm = mixture.GMM(args.num_clusters, n_iter=args.iterations,
                          params=args.gmm_update, init_params='wc')
        gmm.means_ = vocabulary.reshape(args.num_clusters,-1)
        gmm.fit(descriptors)
        vocabulary = gmm

    if args.predict:
        pred = vocabulary.predict(descriptors)
        pred_prob = None
        if 'predict_proba' in dir(vocabulary):
            pred_prob = vocabulary.predict_proba(descriptors)
        for i,f in enumerate(files):
            if pred_prob:
                print '{}\t[{}], ([{}])'.format(os.path.basename(f),
                            pred[i], pred_prob[i])
            else:
                print '{}\t[{}]'.format(os.path.basename(f),
                            pred[i])


    # save gmm
    voc_filepath = os.path.join(args.outputfolder, 
                                (args.vocabulary_filename if
                                 args.vocabulary_filename != None else
                                 args.method) + 'pkl.gz')
    with gzip.open(voc_filepath, 'wb') as f:
        cPickle.dump(vocabulary, f, -1)
    print 'saved vocabulary at', voc_filepath

    if args.method == 'gmm':
        try:
            aic = vocabulary.aic(descriptors)
            print 'aic:', aic
            with open(os.path.join(args.outputfolder, 'aic.txt'), 'a') as f:
                f.write('{}\n'.format(aic))
        except:
            raise
#            print('couldnt compute aic, error: {}'.format(e))


    return os.path.abspath(voc_filepath)

if __name__ == '__main__':
    prep = preprocess.Preprocess.fromArgs()
    parser = argparse.ArgumentParser(description='Clustering - Create'
                                     ' vocabulary',
                                    parents=[prep.parser])
    parser = pc.commonArguments(parser)
    parser = addArguments(parser)
    args = parser.parse_args()

    run(args, prep)
