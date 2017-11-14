import copy
import gzip
import cPickle
from sklearn import preprocessing, decomposition, cluster, linear_model,\
        covariance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from scipy import cluster as sc_cluster
from scipy import linalg as sc_linalg
import numpy as np
import os
import argparse
import cv2
import sys
import progressbar
from zca import ZCA
from rpca import RegularizedPCA

import puhma_common as pc


class Preprocess(object):
    """
    preprocess features
    """
    pca = None
    mean = None
    normalize = []
    trafos = []
    normalize_post = []
    decomp_method = None
    is_centered = False
    n_decomp = 100

    def __init__(self, args=None):
        if args != None:
            vars(self).update(vars(args))

    @staticmethod
    def normalizeData(data, method, aug=None, copy=False):

        if 'l1' in method or 'l1g' in method:
            features = preprocessing.normalize(features, norm='l1', copy=copy)
        if 'l2' in method or 'l2g' in method:
            features = preprocessing.normalize(features, norm='l2', copy=copy)

        if 'hyster' in method or 'hysteresis' in method:
            features = preprocessing.normalize(features, norm='l2', copy=copy)
            features[ features > 0.2 ] = 0.2
            features = preprocessing.normalize(features, norm='l2', copy=copy)

        if 'sift_scale' in method:
            #features = np.min(512.*features, 255.)
            features = 512.0 * features
            features[ features > 255.0 ] = 255.0
        # if power == 0.5 its the same as ssr
        for nm in method:
            if nm.startswith('power'):
                power = float(nm.replace('power', ''))
                features = np.sign(features) * np.abs(features)**power
                break
        if 'ssr' in method:
            # this is actually not needed due to abs...
            #features += np.finfo(float).eps * np.finfo(float).eps
            features = np.sign(features) * np.sqrt(np.abs(features))
        if 'sqrtl2' in method:
            # used as normalization for gaussian(ized) supervectors
            norms = np.linalg.norm(features, 2, axis=1)
            features /= np.sqrt(norms) + np.finfo(float).eps
        # hellinger normalization if vector is already l2 normalized
        # this is the usual case, since SIFT is typically hysteresis-thresholded
        # else choose the 'hellinger_l2' variant
        if 'hellinger' in method:
            # L1 normalization
            features = preprocessing.normalize(features, norm='l1', copy=copy)
            #features += np.finfo(np.float32).eps
            #features /= np.sum(features, axis=1)[:,np.newaxis]
            # square root
            features = np.sign(features) * np.sqrt( np.abs(features) )
        if 'hellinger_l2' in method:
            features = np.sign(features) * np.sqrt( np.abs(features) )
            features = preprocessing.normalize(features, norm='l2', copy=copy)

        if 'sum' in method:
            features = np.sum(features, axis=0).reshape(1,-1)

        if 'l2g2' in method:
            features = preprocessing.normalize(features, norm='l2', copy=copy)

        return data

    def fit_transform(self, features):
        self.fit(features)
        return self.transform(features)


    def decompose(self,features,labels=None):
        if self.pca_components == 0 \
           or self.pca_components+self.start_component > features.shape[1]:
            print ('WARNING no / too many pca-components given, take all'
                    ' (={} dimensions)'.format(features.shape[1]))
            self.pca_components = features.shape[1]
            if self.start_component:
                self.pca_components -= self.start_component

        print ('run {} w. {} components and reg of '
               '{} [start component:{}]'.format(self.decomp_method,
                                                self.pca_components,
                                                self.reg,
                                                self.start_component)\
               + (' + whiten' if self.pca_whiten else ''))

        if 'pca' in self.decomp_method and not 'rpca' in self.decomp_method:
            if features.shape[1] > 500 and features.shape[0] > 500:
                print ('your data seems to be too much ({}) for simple PCA --> '
                       ' try RandomizePCA , however you should run it on asubset and transform then the'
                       ' rest'.format(features.shape))
                self.pca = decomposition.RandomizedPCA(self.pca_components,
                                                       iterated_power=5,
                                                       whiten=self.pca_whiten)
            else:
                self.pca = decomposition.PCA(self.pca_components,
                                             whiten=self.pca_whiten)
        elif 'rpca' in self.decomp_method:
            self.pca = RegularizedPCA(self.pca_components,
                                      whiten=self.pca_whiten,
                                      regularization=self.reg,
                                      start_component=self.start_component)

    def fit(self, features, labels=None):
        print '> preprocess fit'
        if self.aug:
            features = features[:,:-2 if self.aug == 1 else -5]
        elif self.strip_aug:
            features = features[:,:-2 if self.strip_aug == 1 else -5]

        # this is thought for LCS ?
        if self.decomp_method is not None:
            assert(not self.pca)
            # multiple decompositions
            if self.decomp_method.startswith('multi'):
                n_decomp = self.n_decomp
                if self.pca_components > 0:
                    self.pca_components /= n_decomp
                blocksize = int(features.shape[1] / n_decomp)
                all_decomps = []
                for i in range(n_decomp):
                    self.decompose(features[i*blocksize:(i+1)*blocksize],
                                   labels=labels)
                    all_decomps.append(copy.deepcopy(self.pca))
                self.pca = all_decomps

            else:
                self.decompose(features, labels=labels)

   
    def transform(self, features):
        """
        transform features according to selected preprocessing methods
        """
        if len(self.normalize) > 0:
            print ('WARNING: self.normalize is deprecated, add all as '
                  'trafos')
            self.trafos.extend(self.normalize)

        if len(self.trafos) == 0:
            return features
  
        for trafo in self.trafos:
            if trafo == 'pca':
                if self.pca is None: self.load_trafos('pca')
                features = self.pca.transform(features)
                if self.decomp_method == 'pca_zca':
                    # zca-step: rotate data back:
                    features = np.dot(features, self.pca.components_)
            elif 'norm_' in trafo:
                norm = trafo.replace('norm_','')
                features = self.normalizeData(features, [norm], self.aug)
            else:
                raise ValueError('trafo not known')

        if self.aug:
            features = np.concatenate( [features, coords], axis=1 )

        return features

    def load_trafos(self, what):
        if isinstance(what, basestring):
            what = [what]

        if self.load_dir is None:
            raise ValueError('load_dir is None - missing trafo? e.g. fit_prep...')

        for w in what:
            filepath = os.path.join(self.load_dir, w)

            if not filepath.endswith('.pkl.gz'):
                filepath += '.pkl.gz'
            if not os.path.exists(filepath):
                raise ValueError('{} does not exist, cannot load'
                                 ' trafo'.format(filepath))
            with gzip.open(filepath, 'rb') as f:
                stuff = cPickle.load(f)
            self.__dict__[w] = stuff
            print 'loaded', w

        if hasattr(self, 'scaler') and self.scaler:
            assert(isinstance(self.scaler, preprocessing.StandardScaler))
        if hasattr(self, 'pca') and self.pca:
            assert(isinstance(self.pca, decomposition.PCA) \
                   or isinstance(self.pca, RegularizedPCA)\
                  )
            

    def save_trafos(self, outputfolder, verbose=True):
        # everything what could have been fit can also be saved
        attr_to_save = ['pca']
        for attr in attr_to_save:
            if hasattr(self, attr) and self.__getattribute__(attr) is not None:
                filepath = os.path.join(outputfolder, attr + '.pkl.gz')
                pc.dump(filepath, self.__dict__[attr], verbose)

    @classmethod
    def fromArgs(cls):
        parser = argparse.ArgumentParser('preprocess options',
                                         add_help=False)
        cls.parser = addArguments(parser)
        cls.parser.parse_known_args(namespace=cls)

        return cls()

def run(args):
    prep = Preprocess(args)
    runHelper(prep, args)
    return prep

def runHelper(prep, args):

    if not os.path.exists(args.outputfolder):
        pc.mkdir_p(args.outputfolder)

    files, labels = pc.getFiles(args.inputfolder, args.suffix,
                                labelfile=args.labelfile, exact=args.exact,
                                inputfolders_suffix=args.inputfolders_suffix,
                               max_files=args.max_files)
    print 'process {} files'.format(len(files))
    widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
               progressbar.ETA()]


    if args.load_all_features:
        cur_data, index_list = pc.loadDescriptors(files,
                                                  max_descs=args.max_descriptors[0]\
                                                  if args.max_descriptors\
                                                  else 0,
                                                  return_index_list=True)

        # per descriptor labels:
        if len(index_list)-1 != len(labels):
            raise ValueError('{} != {} + 1'.format(len(index_list),
                                                   len(labels)))
        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(labels)
        desc_labels = np.zeros( len(cur_data), dtype=np.uint32)
        for r in xrange(len(labels)):
            desc_labels[index_list[r]:index_list[r+1]] = labels[r]

        print 'loaded all', cur_data.shape
        if 'transform' in args.mode and args.mode != 'fit_transform':
            print 'first feature before:', cur_data[0]
            print 'dimension before:', cur_data.shape[1], cur_data.dtype
            cur_data = prep.transform(cur_data)
            print 'first feature after:', cur_data[0]
            print 'dimension after:', cur_data.shape[1], cur_data.dtype

        if 'fit' in args.mode:
            if 'transform' in args.mode and args.strip_aug:
                prep.strip_aug = False
            prep.fit(cur_data, labels=desc_labels)

            if args.mode == 'fit_transform':
                cur_data = prep.transform(cur_data)

    else:
        progress = progressbar.ProgressBar(widgets=widgets,
                                       maxval=len(files))

        if any(isinstance(f, tuple) for f in files):
            files1 = [f for f in zip(*files)[0]]
            cp = os.path.commonprefix(files1)
        else:
            cp = os.path.commonprefix(files)

        def proj(i):
            # n_samples x n_features
            if not isinstance(args.inputfolder, basestring) and \
               len(args.inputfolder) > 1 or args.inputfolders_suffix != '':
                cur_data = pc.loadMultipleDescriptors(files[i])
                if i == 0:
                    print 'loaded descs of', files[i]
                    print 'shape:', cur_data.shape
            else:
               cur_data = pc.loadDescriptors(files[i])

            if args.mode == 'fit':
                prep.partial_fit(cur_data)
                progress.update(i+1)
                return

            else:
                if i == 0:
                    print 'before:'
                    print cur_data[0]
                    print cur_data.shape, cur_data.dtype

                cur_data = prep.transform(cur_data)

                if i == 0:
                    print 'after:'
                    print cur_data[0,0:min(128,cur_data.shape[1])]
                    print cur_data.shape, cur_data.dtype

            fname = files[i] if isinstance(files[i], basestring)\
                    else files[i][0]

            if os.path.isdir(cp):
                fname = os.path.relpath(fname, cp)

            if fname.endswith('.pkl.gz'):
                name = fname.replace('.pkl.gz','')
            else:
                name = os.path.splitext(fname)[0]

            if os.path.isdir(cp):
                pc.mkdir_p(os.path.join(args.outputfolder,
                    os.path.dirname(name)), silent=True)

            name = os.path.join(args.outputfolder, name  + '_pr.pkl.gz')
#            print fname, '-->', name
            with gzip.open(name, 'wb') as F:
                cPickle.dump(cur_data, F, -1)
            progress.update(i+1)

        progress.start()
        # FIXME: np.dot (e.g. used for (R)PCA) doesnt work in parallel atm
#        if args.parallel:
#            pc.parmap(proj, range(len(files)), args.nprocs)
#        else:
        map(proj, range(len(files)))
        progress.finish()

    prep.save_trafos(args.outputfolder)

def addArguments(parser):
    group = parser.add_argument_group('preprocess options')
    group.add_argument('--rbm',
                   default=[],
                   nargs='*',\
                   help=('load trained RBM, will be used to prep_obj.group '
                         'files'))
    group.add_argument('--pca_whiten', action='store_true',\
                   help='whiten the feature data ( = unit variance) via PCA')
    group.add_argument('--decomp_method', choices=['pca', 
                                                   'rpca'],
                       help='method to use for decomposition')
    group.add_argument('--pca_components', type=int, default=0, \
                   help='apply <decomp_method> decomposition with these n'
                   ' components')
    group.add_argument('--reg', type=float, default=0.0,
                       help='use zca/rpca with regularizing the whitening, i.e. add '
                       'eps to the denominator')
    group.add_argument('--load_dir',
                   help='directory to load stuff from')

    group.add_argument('--mode', choices=['transform', 'fit',
                                         'transform_fit', 'fit_transform'],
                                         default='transform',
                        help='application mode')
    group.add_argument('--trafos', nargs='*', default=[],
                      help='transformations applied in the order of appearance'
                       ' for some it has to be fitted first!'
                       ' e.g.: --trafos norm_hyster norm_hellinger pca nprm_l2g')
    group.add_argument('--load_all_features', action='store_true',
                        help='load all features at once')
    return parser

if __name__ == '__main__':
    # create preprocess instance from arguments
    prep = Preprocess.fromArgs()
    # create special arguments for this main method
    parser = argparse.ArgumentParser('preprocess options',
                                     parents=[prep.parser])
    # ... using common arguments
    parser = pc.commonArguments(parser)
    args = parser.parse_args()
    if args.log:
        log = pc.Log(sys.argv, args.outputfolder)

    runHelper(prep, args)

    if args.log:
        log.dump()
