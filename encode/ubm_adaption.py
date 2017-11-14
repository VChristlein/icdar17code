import os
import gzip
import cPickle
import argparse
import glob
import numpy as np
import cv2
import sys
from sklearn import mixture
from sklearn import preprocessing
from sklearn import decomposition
import copy
import time
import progressbar
#import bob # libgif.so is missing
import math
#import pyvole
import ast
import copy

import puhma_common as pc
from encoding import *
import evaluate
import preprocess

def addArguments(parser):
    ubm_group = parser.add_argument_group('adaptation options')
    ubm_group.add_argument('--load_ubm',\
                        help='filepath to pkl.gz file which contains the ubm-gmm')
    ubm_group.add_argument('--load_gmm',\
                        help=('path to folder containing the scribe-gmms of the '
                              'form: scribe_<nr>_gmm.pkl.gz, e.g. scribe_193_gmm.pkl.gz'))
    ubm_group.add_argument('--save_gmm',action='store_true',\
                        help=('save gmm under outputfolder w. form as in '
                              '--load_gmm'))
    ubm_group.add_argument('--load_scores', '--load_encodings',\
                        help='load score-file (pkl.gz)')
    ubm_group.add_argument('--na', '--no-accumulate',
                           dest='accumulate',
                           action='store_false',
                           help='do not accumulate encodings')
    ubm_group.add_argument('--nbest', type=int, default=0,
                           help='use only the n best mixtures')
    ubm_group.add_argument('--encoding',
                        help='different encoding schemes of the gmm')
    ubm_group.add_argument('--splits', type=int, default=1,\
                        help='data will be split in this #splits (deprecated)')

    ubm_group.add_argument('--update', default='wmc',\
                        help='what to update w. GMM, w:weights, m:means, c:covars')
    ubm_group.add_argument('--concat', action='store_true',
                          help='concatenate all inputfolders')
    ubm_group.add_argument('--concat_later', action='store_true',
                        help='concatenate encodings by filename')

    ubm_group.add_argument('--evaluate', nargs='*', choices=['nothing', 'llr', 'llr_linear',
                                                          'llr_fast', 'cosine', 'pca',
                                                          'emd-m', 'emd-theta',
                                                          'canberra',
                                                          'euclidean', 'dot'
                                                         ],
                        default=['cosine'],
                        help='evaluate with different methods')
    ubm_group.add_argument('--normalize_enc', nargs='*', default=[],
               help='normalization options, note hellinger = l1 '
                   'followed by ssr')
    ubm_group.add_argument('--ratio', default=1.0, type=float,
                           help='filter descriptors which ratio first/second'
                           ' doesnt fit')

    return parser


def loadGMM(filename, lib=''):
    if not os.path.exists(filename):
        raise ValueError('WARNING: gmm doesnt exist')

    if 'bob' in filename:
        gmm = bob.machine.GMMMachine()
        f = bob.io.HDF5File(filename, 'r')
        gmm.load(f)
    else:
        with gzip.open(filename, 'rb') as f:
            gmm = cPickle.load(f)
            if hasattr(gmm, 'weights_') and gmm.weights_.ndim > 1:
                gmm.weights_ = gmm.weights_.flatten()

    # if chosen to use bob-library but the gmm doesn't
    # stem from a bob-gmm then convert it
    if 'bob' not in filename and 'bob' in lib:
        gmm_ = bob.machine.GMMMachine(len(gmm.weights_), gmm.means_.shape[1] )
        gmm_.means = gmm.means_.astype(np.float64)
        gmm_.weights = gmm.weights_.astype(np.float64)
        gmm_.variances = gmm.covars_.astype(np.float64)
        gmm = gmm_

    if hasattr(gmm, 'weights_')\
        and  np.isnan(gmm.weights_).any():
        raise ValueError('nan in weights!')
    if hasattr(gmm, 'means_')\
        and np.isnan(gmm.means_).any():
        raise ValueError('nan in means!')
    if hasattr(gmm, 'covars_')\
        and np.isnan(gmm.covars_).any():
        raise ValueError('nan in covars!')
    if hasattr(gmm, 'skew_')\
        and np.isnan(gmm.skew_).any():
        raise ValueError('nan in skew!')

    if hasattr(gmm, 'weights_')\
        and np.isinf(gmm.weights_).any():
        raise ValueError('inf in weights!')
    if hasattr(gmm, 'means_')\
        and np.isinf(gmm.means_).any():
        raise ValueError('inf in means!')
    if hasattr(gmm, 'covars_')\
        and np.isinf(gmm.covars_).any():
        raise ValueError('inf in covars!')
    if hasattr(gmm, 'skew_')\
        and np.isinf(gmm.skew_).any():
        raise ValueError('inf in skew!')

    # deprecated
    if not hasattr(gmm, 'means_'):
        if np.isinf(gmm).any():
            raise ValueError('inf in your clusters')
        if np.isnan(gmm).any():
            raise ValueError('nan in your clusters')

    return gmm

def run(args, prep=None):
    if prep is None:
        prep = preprocess.Preprocess()

    if not args.labelfile or not args.inputfolder \
       or not args.outputfolder:
        print('WARNING: no labelfile or no inputfolder'
              ' or no outputfolder specified')

    print 'accumulate features:', args.accumulate

    if args.outputfolder and not os.path.exists(args.outputfolder):
        print 'outputfolder doesnt exist -> create'
        pc.mkdir_p(args.outputfolder)

    if args.load_scores:
        print 'try to load computed encodings'


    #####
    # UBM / loading
    print 'load gmm from', args.load_ubm
    ubm_gmm = None
    if args.load_ubm:
        ubm_gmm = loadGMM(args.load_ubm, args.lib)

    #####
    # Enrollment
    # now for each feature-set adapt a gmm
    #####
    if args.labelfile is None:
        print 'WARNING: no label-file'
    if args.concat_later:
        args.concat = True
    if args.concat:
        groups = None

        if args.group_word:
            descriptor_files = pc.getFilesGrouped(args.inputfolder, args.suffix)
            labels = None
        else:
            descriptor_files, labels = pc.getFiles(args.inputfolder, args.suffix,
                                       args.labelfile, exact=False,
                                       concat=True)
            print 'labels:', labels[0]
            if len(descriptor_files) != len(labels):
                raise ValueError('len(descriptor_files) {} !='
                             'len(labels) {}'.format(len(descriptor_files),
                                                 len(labels)))
        print 'num descr-files of first:', len(descriptor_files[0])

    else:
        descriptor_files, labels = pc.getFiles(args.inputfolder, args.suffix,
                                               args.labelfile)
    if args.maskfolder:
        maskfiles = pc.getMaskFiles(descriptor_files, args.suffix, args.maskfolder,
                                args.masksuffix)
    if len(descriptor_files) == 0:
        print 'no descriptor_files'
        sys.exit(1)
    if labels:
        num_scribes = len(list(set(labels)))
    else:
        num_scribes = 'unknown'

    num_descr = len(descriptor_files)
    print 'number of classes:', num_scribes
    print 'number of descriptor_files:', num_descr
    print 'adapt training-features to create individual scribe-gmms (or load saved ones)'
    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
               progressbar.ETA()]
    progress = progressbar.ProgressBar(widgets=widgets,
                                       maxval=len(descriptor_files))

    if 'supervector' in args.encoding:
        identifier = '_sv'
    elif 'fisher' in args.encoding:
        identifier = '_fv'
    else:
        identifier = '_' + args.encoding

    identifier += '_' + args.update
    if len(args.normalize_enc) > 0:
        identifier += '_' + '_'.join(args.normalize_enc)

    encoder = Encoding(args.encoding, ubm_gmm, parallel=False,
                       normalize=args.normalize_enc, update=args.update,
                       relevance=args.relevance, nbest=args.nbest,
                       ratio=args.ratio,
                       accumulate=args.accumulate,
                       nprocs=args.nprocs)

    if args.posteriors_dir:
        posterior_files, _ = pc.getFiles(args.posteriors_dir, args.posteriors_suffix,
                                         args.labelfile)
        print len(posterior_files), len(descriptor_files)
        assert(len(posterior_files) == len(descriptor_files))

    cp = os.path.commonprefix(descriptor_files)
    #print cp
    def encode(i):
        if isinstance(descriptor_files[i], basestring):
            fname = descriptor_files[i]
            if os.path.isdir(cp):
                base = os.path.relpath(fname, cp)

            if fname.endswith('.pkl.gz'):
                base = base.replace('.pkl.gz','')
            else:
                base = os.path.splitext(base)[0]

            if os.path.isdir(cp):
                folder = os.path.join(args.outputfolder,
                    os.path.dirname(base))
                # print 'should create: {} + {}'.format(args.outputfolder, base)
                pc.mkdir_p(folder,silent=True)
        else:
            base = os.path.basename(os.path.commonprefix(descriptor_files[i]))

        gmm_name = base + ('_gmm.pkl.gz' if not 'bob' in args.lib else '_gmm_bob.hdf5')
        gmm = ubm_gmm

        scribe_gmm = None
        # load gmm if possible
        if args.load_gmm:
            gmm_file = os.path.join(args.load_gmm, gmm_name)
            scribe_gmm = load_gmm(gmm_file, args.lib)

        # load encoding
        if args.load_scores:
            if args.load_scores == 'outputfolder':
                load_f = args.outputfolder
            else:
                load_f = args.load_scores

            filepath = os.path.join(load_f, base + identifier + '.pkl.gz')
            if os.path.exists(filepath):
                with gzip.open(filepath, 'rb') as f:
                    enc = cPickle.load(f)
                    return enc, None
#            else:
#                print ('WARNING: encoding {} doesnt exist, compute'
#                        'it'.format(filepath ))


        if args.concat_later:
            enc = []
            for k in range(len(descriptor_files[i])):
                # load data and preprocess
                features = pc.loadDescriptors( descriptor_files[i][k],
                                      min_descs_per_file=args.min_descs, show_progress=(False if\
                                                                  args.concat else True))
                if features is None:
                    print 'features==None'
                    continue
                features = prep.transform(feature)

                enc_ = encoder.encode(features)
                enc.append(enc_)
            enc = np.concatenate(enc, axis=0)

        else:
            # load data and preprocess
            features = pc.loadDescriptors( descriptor_files[i],
                                          min_descs_per_file=args.min_descs,
                                          show_progress=(False if\
                                                         args.concat else
                                                         True)#,
                                         )
            posteriors = None
            if args.posteriors_dir:
                posteriors = pc.loadDescriptors( posterior_files[i] )
                assert(len(posteriors) == len(features))
            if not isinstance(features, np.ndarray) and not features:
                print 'features==None?'
                progress.update(i+1)
                return 0.0, None

            if i == 0:
                print '0-shape:',features.shape
            features = prep.transform(features)
            if i == 0:
                print '0-shape (possibly after pca):',features.shape

            if args.maskfolder:
                sample_weights = pc.loadDescriptors(maskfiles[i])
            else:
                sample_weights = None
            enc, scribe_gmm = encoder.encode(features, return_gmm=True,
                                             sample_weights=sample_weights,
                                             posteriors=posteriors,
                                             verbose=True if i == 0 else False)
            if i == 0:
                print '0-enc-shape', enc.shape
                if isinstance(sample_weights, np.ndarray):
                    print 'sample-weights shape:', sample_weights.shape
            # write
            if args.save_gmm:
                scribe_gmm_filename = os.path.join(args.outputfolder, gmm_name)
                if 'bob' in args.lib:
                    scribe_gmm.save( bob.io.HDF5File(scribe_gmm_filename, 'w') )
                else:
                    with gzip.open(scribe_gmm_filename, 'wb') as f:
                        cPickle.dump(scribe_gmm, f, -1)
                pc.verboseprint('wrote', scribe_gmm_filename)
                progress.update(i+1)

        if args.pq and args.load_pq:
            enc = prep.compress(enc, aug=args.aug)

        # save encoding
        filepath = os.path.join(args.outputfolder,
                                base + identifier + ('_pq' if\
                                args.pq else '') + '.pkl.gz')
        with gzip.open(filepath, 'wb') as f:
            cPickle.dump(enc, f, -1)

        progress.update(i+1)
        if 'nothing' in args.evaluate:
            return None, None
        return enc, scribe_gmm

    progress.start()
    if args.parallel:
        all_enc, all_gmms = zip( *pc.parmap( encode, range(num_descr),
                                            args.nprocs, size=num_descr) )
    else:
        all_enc, all_gmms = zip( *map( encode, range(num_descr) ) )
    progress.finish()
    if 'nothing' in args.evaluate:
        print 'nothing to evaluate, exit now'
        return

    print 'got {} encodings'.format(len(all_enc))

    all_enc = np.concatenate(all_enc, axis=0) #.astype(np.float32)

    print 'all_enc.shape', all_enc.shape

    print 'Evaluation:'

    stats = None
    ret_matrix = None

    for eval_method in args.evaluate:

        ret_matrix, stats = evaluate.runNN( all_enc, labels, distance=True, histogram=False,
                                               eval_method=eval_method,
                                               parallel=args.parallel,
                                               nprocs=args.nprocs)

        if ret_matrix is None or not isinstance(ret_matrix,np.ndarray):
            print 'WARNING: ret_matrix is None or not instance of np.ndarray'
        else:
            fpath = os.path.join(args.outputfolder, 'dist' + identifier
                                 + '_' + eval_method + '.cvs')
            np.savetxt(fpath, ret_matrix, delimiter=',')
            print 'saved', fpath
        return stats

if __name__ == '__main__':
    import stacktracer
    stacktracer.trace_start("trace.html",interval=5,auto=True)

    prep = preprocess.Preprocess.fromArgs()
    parser = argparse.ArgumentParser(description="UBM Adaption",
                                    parents=[prep.parser])
    parser = pc.commonArguments(parser)
    # own 'class' specific arguments
    parser = addArguments(parser)
    args = parser.parse_args()

    run(args, prep)


