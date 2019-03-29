from sklearn import preprocessing
import copy
import os
import numpy as np
import progressbar
import puhma_common as pc
import encoding
import ubm_adaption
import preprocess

""" fit 'local coordinate system' (Delhumeau et al. "Revisiting VLAD..."
features: all (or subset of all) features from the training set
decomp_method: method for decomposition (PCA, ..)

this can also be used to train the trafo for TriEmb
"""

def addArguments(parser):
    parser.add_argument('--lcs_max_descriptors', type=int, default=150000, 
                        help='number of descriptors for LCS')
    parser.add_argument('--resnorm', action='store_true',
                        help='residual norm or not')
    # options needed for triemb
    parser.add_argument('--no_assignment', action='store_true',
                        help='dont choose descriptors, take all')
    parser.add_argument('--global_cs', action='store_true',
                        help='apply LCS on the global vec, not component-wise')
    # TODO: add ratio for getAss and k
    return parser

def run(args):
    print '> compute LCS'
    files, labels = pc.getFiles(args.inputfolder, args.suffix, args.labelfile,
                        exact=args.exact)
    if len(args.max_descriptors) == 0:
        descriptors, index_list = pc.loadDescriptors(files, rand=True, return_index_list=1)
    else:
        descriptors, index_list = pc.loadDescriptors(files,\
                                         max_descs=args.lcs_max_descriptors,
                                         max_descs_per_file=max(int(args.lcs_max_descriptors/len(files)),\
                                                                1), 
                                         rand=True,
                                        return_index_list=1)
        print 'descriptors.shape', descriptors.shape
#        #if not args.inputfolders:
#        cur_data, index_list = pc.loadDescriptors(files,
#                                                  max_descs=args.max_descriptors[0]\
#                                                  if args.max_descriptors\
#                                                  else 0,
#                                                  return_index_list=True)

    # per descriptor labels:
    if len(index_list)-1 != len(labels):
        raise ValueError('{} != {} + 1'.format(len(index_list),
                                               len(labels)))
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    desc_labels = np.zeros( len(descriptors), dtype=np.uint32)
    for r in xrange(len(labels)):
        desc_labels[index_list[r]:index_list[r+1]] = labels[r]


    prep = preprocess.Preprocess(args)

    ubm = ubm_adaption.loadGMM(args.load_ubm)
    if not args.no_assignment:
        assignments = encoding.getAssignment(ubm.means_, descriptors)
    lcs = []
    descr = []
    # Note: we could also compute the LCS afterwards using 'multipca' option 
    # of preprocess...
    for i in range(len(ubm.means_)):
        if args.no_assignment:
            diff = descriptors - ubm.means_[i]
        else:
            for_lcs = descriptors[ assignments[:,i] > 0 ]
            diff = for_lcs - ubm.means_[i]
        if args.resnorm:
            diff = preprocessing.normalize(diff, norm='l2', copy=False) 
        if not args.global_cs:
            prep.fit(diff, desc_labels[assignments[:,i] > 0])
            lcs.append(copy.deepcopy(prep.pca))
            prep.pca = None
        else:
            descr.append(diff)

    if args.global_cs:
        print '> compute global lcs'
        diff = np.concatenate(descr, axis=1)
        print '... from descr.shape', diff.shape
        prep.fit(diff, desc_labels)
        print '< compute global lcs'
        lcs = copy.deepcopy(prep.pca)
        prep.pca = None
    folder = os.path.join(args.outputfolder, 'lcs.pkl.gz')
    pc.dump(folder, lcs)
    return folder
