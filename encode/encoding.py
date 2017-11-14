import cv2
import numpy as np
from sklearn import preprocessing, mixture, cluster
import puhma_common as pc
import math
#import sys
import time
import copy
import preprocess
import extract_ivector, compute_bw_stats

class DummyGMM(object):
    def __init__(self, weights, means, covars):
        self.weights_ = weights
        self.means_ = means
        self.covars_ = covars

class Encoding(object):
    # TODO: make it completely object-oriented 
    """ 
    wrapper class around encoding methods
    """
    def __init__(self, method, ubm, 
                 parallel=False, 
                 normalize=['l2g','l2c'], 
                 update='wmc', 
                 relevance=28, 
                 nbest=0,
		 ratio=1.0, 
                 posterior_theta=0.0,
                 accumulate=True,
                 nprocs=None):
        self.method = method
        self.ubm = ubm
        self.parallel = parallel
        self.normalize = normalize
        self.update = update
        self.relevance = relevance
        self.nbest = nbest
	self.ratio = ratio
        self.posterior_theta = posterior_theta
        self.accumulate = accumulate
        self.nprocs = nprocs
        
        if not hasattr(ubm, 'type') and not hasattr(ubm, 'means_'):
            self.ttype = 'kmeans'
            self.means = ubm
        else:
            self.ttype = ubm.type_
            self.means = ubm.means_


    def encode(self, features, posteriors=None, return_gmm=False,
               sample_weights=None, center_indices=None, verbose=False): 
#        if method == 'gnostic:'
#            enc = gnostic(features, posteriors)
#        else:
        enc, gmm = encodeGMM(self.method, self.ubm, features, 
                             parallel=self.parallel, 
                             normalize=self.normalize, 
                             relevance=self.relevance,
                             update=self.update, 
                             posteriors=posteriors,
                             accumulate=self.accumulate, 
                             posterior_theta=self.posterior_theta,
                             posterior_nbest=self.nbest,
                             ratio = self.ratio,
                             sample_weights=sample_weights,
                             nprocs=self.nprocs, verbose=verbose)

        if return_gmm:
            return enc, gmm
        return enc

def getAssignment(means, data, ratio=1.0):
    """
    return assignments for given means and data, i.e. search for each datapoint 
        the nearest cluster and set this cluster-index to 1
        ratio:  filter assignments by ratio, take only those where 1st / 2nd
                distances <= ratio 
    """
    # get nearest matches
    # slower:
#    trainer = cluster.KMeans(len(means))
#    trainer.cluster_centers_ = means
#    center_idx = trainer.predict(data)

    # TODO: allow k > 1 
    # -> make center_idx_ a matrix and add for-loops
    center_idx_ = []
    # equivalent but faster:    
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(data.astype(np.float32), 
                               means.astype(np.float32),
                               k = 1 if ratio >= 1.0 else 2)

    if ratio >= 1.0:
        for m in matches:
            center_idx_.append( m[0].trainIdx )
    else:
        for m in matches:
            if m[0].distance / m[1].distance <= ratio:
                center_idx_.append( m[0].trainIdx )

    center_idx = np.array(center_idx_)

    # create hard assignment
    #assignment = np.zeros( (len(data), len(means)), dtype=np.uint8)
    assignment = np.zeros( (len(data), len(means)) )
    for i in range(len(data)):
        assignment[ i, center_idx[i] ] = 1
   
    return assignment

def nBest(posteriors, nbest=0):
    """
    nbest: keep only the nbest, i.e. n highest posteriors
    """
    # create grid
    gr = np.ogrid[:posteriors.shape[0], :posteriors.shape[1]]        
    # partial  sorting
    indices = np.argpartition(-posteriors, nbest, axis=1)
    # replace axis 1 with all but the nbest 
    gr[1] = indices[:,nbest:]        
    # ... and set them to 0
    posteriors[gr] = 0.0

    # and re-normalize them such that all posteriors sum
    # up to 1 again
    posteriors = preprocessing.normalize(posteriors, norm='l1', copy=False)

    return posteriors

def getPosteriors(gmm, data, 
                  parallel=None, 
                  theta=0.0, 
                  hard_assignment=False,
                  nprocs=None, 
                  ratio=1.0):
    """
    compute the posterior probability (assignment) for each sample

    parameters:
        gmm: scikit-learn computed gmm
        data: feature-vectors row-wise
        parallel: if true it will be computed in parallel
        theta: posterior threshold, i.e. if eps > 0.0 each posterior < eps will be
            set to 0 Sanchez et al. use here 1e-4
        hard_assignment: if set to true, then 'getAssignment is called with the
            gmm's means -> much faster than predicting the poseriors
    """
    if hard_assignment:
        return getAssignment(gmm.means_, data, ratio)

    if parallel:
        def predict(split):
            return gmm.predict_proba(split)
        splits = np.array_split(data, 8) 
        posteriors = pc.parmap(predict, splits, nprocs)
        posteriors = np.concatenate(posteriors, axis=0)
    else:
        posteriors = gmm.predict_proba(data)

    if theta > 0.0:
        # set all posteriors smaller eps to 0
        posteriors[ posteriors < theta ] = 0.0
        # re-normalize the posteriors such that they sum up to 1 again
        posteriors = preprocessing.normalize(posteriors, norm='l1', copy=False)

    return posteriors

def encodeGMM(method, gmm, data, parallel=False, 
              normalize=['ssr', 'l2g'], relevance=28, update='wmc',
              posteriors=None, accumulate=True, posterior_theta=0.0,
              posterior_nbest=0, ratio=1.0, sample_weights=None,
              nprocs=None, verbose=False):
    """
    Encoding scheme adapting data from a GMM

    parameters:
        method: an encoding method to call, currently acceptable
            'supervector', 'fisher', 'vlad', 'gaussiansv' 
        gmm: the background gmm = vocabulary = universal background model
        data: typiclly the features, row wise order
        parallel:  set it to True if you want partially parallelized code
        normalize: global and local normalization schemes
        relevance: (gmm-supervector-specific): relevance factor for mixing 
        update: (gmm-supervector-specific): which parts to use for mixing 
        posteriors: if given, they won't be computed again
        accumulate: if true, calculate one encoding for all features, else
                    one for each feature (be aware of huge memory drive cost)
    returns: encoded vector and reference to gmm / adapted gmm if
                'supervector'
    """

    if not hasattr(gmm, 'type') and not hasattr(gmm, 'means_'):
        means = gmm
        ttype = 'kmeans'
    else:
        means = gmm.means_
        ttype = gmm.type_

    # sklearn 1.18 changed the name...
    if hasattr(gmm, 'covariances_'):
        gmm.covars_ = gmm.covariances_

    #if posteriors is None and not 'gmp' in method 
    if posteriors is None and not 'tr' in method and not 'gmp' in method \
        and not 'sum' in method:
        # compute posteriors from gmm
        if 'bowhard' == method or ttype == 'kmeans' or\
           method == 'vlad':
            posteriors = getAssignment(means, data, ratio)
        else:
            posteriors = getPosteriors(gmm, data, parallel, 
                                       theta=posterior_theta,
                                       nprocs=nprocs)
    if posterior_nbest > 0:
        posteriors = nBest(posteriors, posterior_nbest)

    if isinstance(sample_weights, np.ndarray):
        # easiest way to weight the samples is by modifying the posteriors
        posteriors *= sample_weights.reshape(-1,1)
   
    # check if we actually use gmp
    if accumulate == True:
        for nm in normalize:
            if 'gmp' in nm:
                accumulate=False
                print 'set accumualte to False, since GMP is used'
                break

    if 'fisher' == method or 'fv' == method:
        enc = fisherEncode(gmm, data, 
                           posteriors, parallel, normalize, 
                           accumulate=accumulate,
                           fv_components=update)
    elif method.startswith('vlad'):
        enc = vladHard(data, means, posteriors, parallel, means.shape[0], 
                       normalize, lcs=lcs)
    # probabilistic vlad
    elif 'pvlad' in method:
        enc = vlad(data, means, posteriors, parallel, means.shape[0], normalize )            
    elif 'bow' in method:
        enc = np.sum(posteriors, axis=0)
    elif 'sum' in method:
        enc = np.sum(data, axis=0).reshape(1,-1)
    else:
        raise ValueError('unknown encoding method {}'.format(method)) 
   
#    pc.check(enc, 'encoding')
    
    if verbose:
        print("> after encoding, before normalizeData()")
        print(enc.shape)
        print(enc)
    norm_enc = preprocess.Preprocess.normalizeData(enc, normalize, copy=True)    
    if verbose:
        print("> after encoding, after normalizeData()")
        print(norm_enc.shape)
        print(norm_enc)
    
    return norm_enc, gmm

def fisherEncode( gmm, data, posteriors, 
                  parallel=False, 
                  normalize=[], 
                  accumulate=True,
                 fv_components='wmc'):

#    if not np.all( np.isfinite(gmm.covars_) ):
#        print 'WARNING: gmm-covars contains (+/-)inf/nan values'
#    if np.any( gmm.covars_ < 0 ):
#        print 'WARNING: gmm-covars contains negative values'

    if gmm.covars_.ndim == 3:
        wk_, uk_, vk_ = fisherFull( data, gmm.means_, gmm.covars_, gmm.weights_,
                        posteriors, parallel, accumulate=accumulate)
    else:
        inv_sqrt_cov = np.sqrt(1.0 / (gmm.covars_ + np.finfo(np.float32).eps))
        wk_, uk_, vk_ = fisherCPU( data, gmm.means_, gmm.weights_,
                        posteriors, inv_sqrt_cov, parallel,
                        accumulate=accumulate, normalize=normalize,
                                  update=fv_components)

    components, fd = gmm.means_.shape

    wk_ = np.array(wk_)
    uk_ = np.array(uk_)
    vk_ = np.array(vk_)

    if not accumulate:
        if 'w' in fv_components:
            wk_ = wk_.T
        if 'm' in fv_components:
            uk_ = np.transpose(uk_, (1, 0, 2))
        if 'c' in fv_components:
            vk_ = np.transpose(vk_, (1, 0, 2))

    # component-wise normalization
    if 'l2c' in normalize: 
        if accumulate:
            if 'm' in fv_components:
                wk_ = preprocessing.normalize(wk_)
            if 'm' in fv_components:
                uk_ = preprocessing.normalize(uk_.reshape(components, fd))
            if 'm' in fv_components:
                vk_ = preprocessing.normalize(vk_.reshape(components, fd))
        else:
            for i in range(len(uk_)):
                if 'm' in fv_components:
                    wk_[i] = preprocessing.normalize(wk_[i])
                if 'm' in fv_components:
                    uk_[i, :, :] = preprocessing.normalize(\
                        uk_[i,:,:].reshape(components, fd))
                if 'm' in fv_components:
                    vk_[i, :, :] = preprocessing.normalize(\
                        vk_[i,:,:].reshape(components, fd))
    # stacking
    enc = []
    if accumulate:
        if 'w' in fv_components: 
            enc.append(wk_.reshape(1,-1))
        if 'm' in fv_components:
            enc.append(uk_.reshape(1,-1))
        if 'c' in fv_components:
            enc.append(vk_.reshape(1,-1))
    else:
        nsamples = uk_.shape[0]
        if 'w' in fv_components: 
            enc.append(wk_.reshape(nsamples,-1))
        if 'm' in fv_components: 
            enc.append(uk_.reshape(nsamples,-1))
        if 'c' in fv_components: 
            enc.append(vk_.reshape(nsamples,-1))
 
    fv = np.concatenate(enc, axis=1) 
    
    return fv

def fisherCPU(data_orig, means, weights, posteriors_orig, 
              inv_sqrt_cov,
              parallel=False,
              accumulate=True,
              normalize=[], update='wmc'):
    
    components, fd = means.shape

    def encode(i):
        data = data_orig[ posteriors_orig[:,i] > 0 ]
        posteriors = posteriors_orig[ posteriors_orig[:,i] > 0, i].reshape(1,-1)
        clustermass = len(data)
        
        diff = (data - means[i]) * inv_sqrt_cov[i]
        if 'rn' in normalize:
            diff = preprocessing.normalize(diff, norm='l2', copy=False) 

        if accumulate:
               #diff = data * inv_sqrt_cov[i]
            if 'w' in update and clustermass > 0:
                weights_ = np.sum(posteriors - weights[i])
                weights_ /= ( len(data) * math.sqrt(weights[i]) )
            else:
                weights_ = 0   

            if 'm' in update and clustermass > 0:
                means_ = posteriors.dot( diff )
                means_ /= ( len(data) * math.sqrt(weights[i]) )
            else:
                means_ = np.zeros( (1,fd), data.dtype)

            if 'c' in update and clustermass > 0:
                covs_ = posteriors.dot( diff*diff - 1 )
                covs_ /= ( len(data) * math.sqrt(2.0*weights[i]) )
            else:
                covs_ = np.zeros( (1,fd), data.dtype)
           
        else:
            if 'w' in update:
                weights_ = posteriors.T - weights[i]
                weights_ /= math.sqrt(weights[i]) 
            else:
                weights_ = None 

            if 'm' in update and clustermass > 0:
                means_ = posteriors.T * diff
                means_ /= math.sqrt(weights[i])
            else:
                means_ = np.zeros( (len(data), fd), data.dtype)

            if 'c' in update and clustermass > 0:
                covs_ = posteriors.T * (diff*diff - 1 )            
                covs_ /= math.sqrt(2.0*weights[i]) 
            else:
                covs_ = np.zeros( (len(data), fd), data.dtype)

#        print 'w:', weights_
#        print 'm:', means_
#        print 'c:', covs_

#        print 'w:', weights_.shape
#        print 'm:', means_.shape
#        print 'c:', covs_.shape


        return weights_, means_, covs_


    if parallel:
        wk_, uk_, vk_ = zip( *pc.parmap(encode, range(components)) )
    else:
        wk_, uk_, vk_ = zip( *map(encode, range(components)) )
    
    return wk_, uk_, vk_

def fisherFull( data, means, covars, weights,
               posteriors, parallel, accumulate=True):

    d = covars.shape[1] 
    indices = np.triu( np.ones((d,d)) ).flatten().astype(np.bool)

    def encode(i):
        inv_cov = np.linalg.inv(covars[i])
        diff = data - means[i]
        
        # compute means
        z = diff.dot(inv_cov)

        # compute covars
        covs = np.zeros( (len(data), d*d), data.dtype)
        for a in range(d):
#            tmp = - z * np.roll(z, a, axis=1) \
#                                     - 0.5 * ( (2*np.pi)**(-d) ) * \
#                                     np.diag( np.roll(inv_cov, -a, axis=0) )
#            print tmp.shape, covs.shape

            covs[ :, a*d:(a+1)*d ] = - z * np.roll(z, a, axis=1) \
                                     - 0.5 * ( (2*np.pi)**(-d) ) * \
                                     np.diag( np.roll(inv_cov, -a, axis=0) )

        # just take the upper triangle matrix
        covs = covs[:,indices]

        # dub in posteriors
        if accumulate:
            weights_ = np.sum(posteriors[:,i] - weights[i])
            means_ = posteriors[:,i].T.dot( z )
            covs_ = posteriors[:,i].T.dot( covs )
        else:
            weights_ = posteriors[:,i] - weights[i]
            means_ = posteriors[:,i].reshape(-1, 1) *  z
            covs_ = posteriors[:,i].reshape(-1, 1) * covs
       
        weights_ /= ( len(data) * math.sqrt(weights[i]) )
        # TODO: Fisher information

        return weights_, means_, covs_
        
    if parallel:
        wk_, uk_, vk_ = zip(*pc.parmap(encode, range(means.shape[0])))
    else:
        wk_, uk_, vk_ = zip( *map(encode, range(means.shape[0])))

    return wk_, uk_, vk_

def vladPure(data, means, assignments, parallel, components,
             normalize=['l2c'], covars=None, skew=None):
    def encode(k):
        vk_ = None
        sk_ = None

        possible = data[ assignments[:,k] > 0 ]
        clustermass = len(possible)
        if clustermass > 0:
            agg = np.sum( possible, axis=0 )
            uk_ = agg - clustermass * means[k]
        else:
            uk_ = np.zeros( data.shape[1], dtype=data.dtype)
        
        if 'l2c' in normalize:
            n = max( math.sqrt( enc.dot(enc) ), 1e-12)
            enc /= n

        return enc

    if parallel:
        uk = pc.parmap(encode, range(components))
    else:
        uk = map(encode, range(components))

    uk = np.concatenate(uk).reshape(1,-1)

    return uk # * assignments.sum()

def vladHard(data, means, assignments, parallel, components,
             normalize=['l2c'], covars=None, skew=None, lcs=None):
    """
    compute 'vector of locally aggregated descriptors'
    for hard assignment only - this way it can be computed faster
    """
    cgmp = False
    for nm in normalize:
        if nm.startswith('cgmp'):
            alpha_str = nm.replace('cgmp', '')
            if alpha_str == '':
                raise ValueError('no alpha for cgmp given')

            alpha = float(alpha_str)
            cgmp = True
            break

    def encode(k):         
        vk_ = None
        sk_ = None

        possible = data[ assignments[:,k] > 0 ]
        clustermass = len(possible)
        if clustermass > 0:
            """
            'rn':
                Delhumeau: Revisiting VLAD ...            
            """
            if 'rn' in normalize:
                diff = possible - means[k]
                if 'rn' in normalize:
                    diff = preprocessing.normalize(diff, norm='l2', copy=False) 
                else:
                    uk_ = np.sum(diff, axis=0)

            else:
                agg = np.sum( possible, axis=0 )
                if 'mass' in normalize:
                    uk_ = agg / clustermass
                    uk_ -= means[k]
                else:
                    uk_ = agg - clustermass * means[k]

        enc = [uk_]
        enc = np.concatenate(enc)

        if 'l2c' in normalize and clustermass > 0:
            enc = preprocessing.normalize(enc, norm='l2', copy=False) 

        return enc

    if parallel:
        uk = pc.parmap(encode, range(components))
    else:
        uk = map(encode, range(components))

    uk = np.concatenate(uk).reshape(1,-1)

    return uk # * assignments.sum()

def vlad(data, means, assignments, parallel, components, 
               normalize=['l2c', 'mass']):
    """
    compute 'vector of locally aggregated descriptors'
    assignments are probabilistically computed
    """
    def encode(k):
        #diff = data - means[k]
        if 'rn' in normalize:
            diff = data - means[k]
            diff = preprocessing.normalize(diff, norm='l2', copy=False) 
            uk_ = assignments[:,k].T.dot(diff)        
        else:
            uk_ = assignments[:,k].T.dot(data)        
            # this is equal to:
    #        uk__ = np.zeros( (1, data.shape[1]), dtype=np.float32)
    #        for i in range(len(data)):
    #            uk__ += assignments[i,k] * data[i]

            clustermass = assignments[:,k].sum()
            if clustermass > 0:
                if 'mass' in normalize:
                    uk_ /= clustermass
                    uk_ -= means[k]
                else:
                    uk_ -= clustermass * means[k]

        if 'l2c' in normalize:
            n = max(math.sqrt(np.sum(uk_ * uk_)), 1e-12)
            uk_ /= n

        return uk_

    if parallel:
        uk = pc.parmap(encode, range(components))
    else:
        uk = map(encode, range(components))

    uk = np.concatenate(uk).reshape(1,-1)

    return uk # * assignments.sum()

if __name__ == '__main__':
    import yaml
    import sys
    import os

    def opencv_matrix(loader, node):
        mapping = loader.construct_mapping(node, deep=True)
        mat = np.array(mapping["data"])
        mat.resize(mapping["rows"], mapping["cols"])
        return mat
    yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)

    # Once you've done that, loading the yaml file is simple:
    with open(sys.argv[1]) as fin:
        fv_desc = yaml.load(fin.read())
        #print fv_desc[3].keys()
        #print len(fv_desc)
        #print fv_desc[1]
        fv = fv_desc['fisher']
        desc = fv_desc['desc']
        print 'desc.shape', desc.shape
        print 'fv.shape', fv.shape

    with open(sys.argv[2]) as fin:
        gmm_d = yaml.load(fin.read())

        gmm = mixture.GMM(192)
        gmm.weights_ = gmm_d['weight'].reshape(-1)    
        gmm.means_ = gmm_d['mean']
        covars = []
        for i in range(192):
            covars.append(np.diag(gmm_d['covs_' + str(i)]).reshape(1,32))
        covars = np.concatenate(covars, axis=0)
        gmm.covars_ = covars
        
        print 'weights.shape:', gmm.weights_.shape
        print 'mean.shape:', gmm.means_.shape
        print 'covars.shape:', covars.shape, ' == 192x32?'

    encoding = Encoding('fisher', gmm, normalize=['ssr', 'l2g'],
#    encoding = Encoding('fisher', gmm, normalize=['hellinger'],
                        update='mc')  
    st = time.time()
    enc = encoding.encode(desc)
    et = time.time() - st
    print 'elapsed time:', et


    print 'sum(fv):', np.sum(np.abs(fv))
    print 'sum(enc):', np.sum(np.abs(enc))

    print 'fvs differ by: ', np.sum(np.abs(enc.astype(np.float32)) -
                                    np.abs(fv.astype(np.float32)))
        
    print fv[0, :10]
    print 'vs'
    print enc[0, :10]

    print '-- Compute vl via wrapper --'        

    import pyvole
#    print desc.dtype, gmm.means_.dtype, gmm.means_.dtype
    st = time.time()
    vl_enc = pyvole.puhma_featureex.vl_fisher_encode(desc.astype(np.float32), 
                                                     gmm.means_.astype(np.float32), 
                                                     gmm.covars_.astype(np.float32),
                                                     gmm.weights_.astype(np.float32))
    et = time.time() - st
    print 'elapsed time:', et
    print 'sum(vl_enc):', np.sum(np.abs(vl_enc))



