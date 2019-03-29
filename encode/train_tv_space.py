import os
import numpy as np
import puhma_common as pc
import compute_bw_stats
import ubm_adaption
import progressbar

# TODO: consider using sidekit's version which is probably much faster

def addArguments(parser):
    parser.add_argument('--load_stats', action='store_true', 
                        help='load zero and first order stats')
    parser.add_argument('--tv_dim', type=int, default=400,
                        help='number of tv dimensions')
    parser.add_argument('--tv_niter', type=int, default=5,
                        help='number of em iterations fot the '
                        'computation of the tv space')
    return parser

def run(args):
    print '> compute tv space'
    files, _ = pc.getFiles(args.inputfolder, args.suffix, args.labelfile,
                        exact=args.exact)
    ubm = ubm_adaption.loadGMM(args.load_ubm)

    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
               progressbar.ETA()]
    progress = progressbar.ProgressBar(widgets=widgets,
                                   maxval=len(files))
    print 'extract stats'
    def extract(i):
        descr = pc.loadDescriptors(files[i])
        of = os.path.join(args.outputfolder,
                        os.path.basename(files[i]).split('.',1)[0]+
                       '_stat.pkl.gz')
        if args.load_stats and os.path.exists(of):
            N, F = pc.load(of)
        else:
            N, F = compute_bw_stats.compute_bw_stats(descr, ubm, None, args.nbest)
            pc.dump(of, [N,F], verbose=False)
        if i == 0:
            print N.shape, F.shape
        progress.update(i+1)
        return N.reshape(1,-1), F.reshape(1,-1)

    progress.start()
    if args.parallel:
        Ns, Fs = zip( *pc.parmap(extract, range(len(files)),
                                 nprocs=args.nprocs))        
    else:
        Ns, Fs = zip( *map(extract, range(len(files))))
    progress.finish()

    Ns = np.concatenate(Ns, axis=0)
    Fs = np.concatenate(Fs, axis=0)
    print 'train tv from {} stats'.format(len(Ns))
    tv = train_tv_space(Ns, Fs, ubm, args.tv_dim, args.tv_niter, args.parallel,
                       args.nprocs)

    folder = os.path.join(args.outputfolder, 'tv.pkl.gz')
    pc.dump(folder, tv)

    return folder

def train_tv_space(N, F, ubm, tv_dim, niter, parallel, nprocs):
    """
    train total-variability matrix
    Assume factor analysis (FA) model of the from:

           M = m + T . x

    for mean supervectors M, the maximum likelihood 
    estimate (MLE) of the factor loading matrix T (aka the total variability 
    subspace) is computed. M is the adapted mean supervector, m is the UBM mean 
    supervector, and x~N(0,I) is a vector of total factors (aka i-vector).% References:
        
    References
    [1] D. Matrouf, N. Scheffer, B. Fauve, J.-F. Bonastre, "A straightforward 
        and efficient implementation of the factor analysis model for speaker 
        verification," in Proc. INTERSPEECH, Antwerp, Belgium, Aug. 2007, 
        pp. 1242-1245.  
    [2] P. Kenny, "A small footprint i-vector extractor," in Proc. Odyssey, 
        The Speaker and Language Recognition Workshop, Singapore, Jun. 2012.
    [3] N. Dehak, P. Kenny, R. Dehak, P. Dumouchel, and P. Ouellet, "Front-end 
        factor analysis for speaker verification," IEEE TASLP, vol. 19, pp. 
        788-798, May 2011. 
    """
    nmix, ndim = ubm.means_.shape
    try: 
        S = ubm.covariances_.T.reshape(-1,1)
    except AttributeError:
        S = ubm.covars_.T.reshape(-1,1)
   
   # random initialization (JFA cookbook)
    T = np.random.randn(tv_dim, nmix*ndim) * np.sum(S, axis=0) * 0.001

    # re-estimating total subspace w. tv_dim factors
    for i in range(niter):
        print 'Train TV-space, iter: {}'.format(i)
        LU, RU = expectation_tv(T, N, F, S, tv_dim, nmix, ndim, parallel, nprocs)
        T = maximization_tv(LU, RU, ndim, nmix)
    return T

def expectation_tv(T, N, F, S, tv_dim, nmix, ndim, parallel, nprocs):
    # compute posterior means and covariance matrices of the factors 
    # = latent variables
    idx_sv = np.arange(nmix).repeat(ndim).reshape(-1)
    nfiles = N.shape[0]

    LU = nmix * [np.zeros((tv_dim, tv_dim))]
    RU = np.zeros((tv_dim, nmix*ndim))
    I = np.eye(tv_dim)
    T_invS = T / S.T 

    # mini-batch 
    #bs = 250 # adjust me
    bs = 400 # adjust me
    nbatch = int( nfiles / float(bs) + 0.999)
    for i in range(nbatch):
        end = min(nfiles, (i+1)*bs)
        N1 = N[ i*bs : end ]
        F1 = F[ i*bs : end ]
        dim = N1.shape[0]
#        Ex = np.zeros((tv_dim, dim))
#        Exx = np.zeros((tv_dim, tv_dim, dim))

        widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
               progressbar.ETA()]
        progress = progressbar.ProgressBar(widgets=widgets,
                                   maxval=dim)

        #for ix in range(dim): 
        def posteriors(ix):
            tmp = T_invS * N1[ix, idx_sv]
            L = I + tmp.dot(T.T)
            
            Cxx = np.linalg.pinv(L) # posterior covariance Cov(x,x)
            B = T_invS.dot(F1[ix].T).reshape(-1,1)
            Ex_ = Cxx.dot(B).reshape(-1,1) # posterior mean E[x]

            Exx_ = Cxx + Ex_.dot(Ex_.T)
            progress.update(ix+1)
            return Ex_.reshape((tv_dim,1)), Exx_.reshape((tv_dim, tv_dim, 1))

        progress.start()
        if parallel:
            Ex, Exx = zip( *pc.parmap(posteriors, range(dim), nprocs=nprocs) )
        else:
            Ex, Exx = zip( *map(posteriors, range(dim)))
        progress.finish()
        
        Ex = np.concatenate(Ex, axis=1)
        Exx = np.concatenate(Exx, axis=2)
       
        RU = RU + Ex.dot(F1)
        # TODO: parallelize me ?
        for mix in range(nmix): 
            tmp = Exx * N1[:, mix].T.reshape(1,1,dim)
            #tmp_m = octave.get_n(Exx, N1, mix+1, dim)
            LU[mix] = LU[mix] + np.sum(tmp, axis=2)

    return LU, RU

def maximization_tv(LU, RU, ndim, nmix):
    """
    ML re-estimation of the total subspace matrix or the factor loading
    matrix
    """
    for i in range(nmix):
        s = i*ndim
        e = (i+1)*ndim
        
        tmp = np.linalg.solve(LU[i], RU[:, s:e])
#        tmp = np.linalg.lstsq(LU[i], RU[:, s:e])[0]
        RU[:, s:e] = tmp
    return RU
