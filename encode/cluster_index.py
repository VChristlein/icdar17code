import puhma_common as pc
import progressbar
from sklearn import cluster, preprocessing
import argparse
import numpy as np
import sys
import os
import cv2
import struct

"""
given
- the patches (--df + --ds -l)
- the descriptors (--inputfolder --suffix -l)
- the clusters, assumes 'means_' attribute

it computes the label for each patch, i.e. the cluster index of 
the nearest cluster to the associated descriptor

param 'ratio' can be used to ignore descr. at the boundary of clusters

output: labels, real-labels (converted to numbers), patches as one big ocvmb
-> use ocvmb2tt to convert these then to torch tensors
"""

def addArguments(parser):
    parser.add_argument('--df', '--descriptor_folder', 
                        nargs='+', 
                        help='actual descriptors to save then')
    parser.add_argument('--ds', '--descriptor_suffix', 
                        help='suffix for descriptor folder')
    parser.add_argument('--cluster', 
                        required=True,
                        help='points to the cluster file')
    parser.add_argument('--ratio', type=float,
                        help='max ratio 1st to 2nd nearest cluster')
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clustering - Index')
    parser = pc.commonArguments(parser)
    parser = addArguments(parser)
    args = parser.parse_args()
    np.random.seed(42)

    if not os.path.exists(args.outputfolder):
        pc.mkdir_p(args.outputfolder)

    print args.max_descriptors
    
    files, labels = pc.getFiles(args.inputfolder, args.suffix, args.labelfile,
                                concat=True)
    print 'n-files:', len(files)
    
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(labels)
    
    desc_files, _ = pc.getFiles(args.df, args.ds,
                                 args.labelfile, concat=True)

    kmeans = pc.load(args.cluster)
    means = kmeans.means_


    print files[0], desc_files[0]
    dummy_desc = pc.loadDescriptors(files[0])
    dummy_desc2 = pc.loadDescriptors(desc_files[0])
    assert(dummy_desc.shape[0] == dummy_desc2.shape[0])

    print 'descr.shape:', dummy_desc.shape
    desc = np.zeros((args.max_descriptors[0], dummy_desc2.shape[1]),
                            dtype=np.float32)
    labels_out = np.zeros( (args.max_descriptors[0],1), dtype=np.float32)
    labels_real = np.zeros( (args.max_descriptors[0],1), dtype=np.float32)
    max_descs_per_file = args.max_descriptors[0] / len(files)

    cluster_idx = []
    i = 0   
    visited_files = {}
    no_new = False
    while i < args.max_descriptors[0]:
        widgets = [ progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA() ]
        progress = progressbar.ProgressBar(widgets=widgets, maxval=len(files))
        progress.start()
        for k, f in enumerate(files):
            data = pc.loadDescriptors(f)
            real_descriptors = pc.loadDescriptors(desc_files[k])
            assert(data.shape[0] == real_descriptors.shape[0])
                        
            indices = np.random.choice(len(data), 
                                       min(len(data),
                                           max_descs_per_file),
                                       replace=False)
            if f in visited_files.keys():
                # get visited indices
                v_ind = visited_files[f]
                # mask those out
                indices = indices[~np.in1d(indices, v_ind)]
                if len(indices) == 0:
                    print 'no new indices: break here'
                    no_new = True
                    break
                visited_files[f] = np.concatenate([v_ind,indices])
            else:
                visited_files[f] = indices

            data = data[ indices ]
            real_descriptors = real_descriptors[indices]

#            print '> got {} descriptors'.format(len(data))
#            print 'match now all'
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            matches = matcher.knnMatch(data.astype(np.float32), 
                                       means.astype(np.float32),
                                       k = 2)
#            print '> finished matching'
            for e,m in enumerate(matches):
                # Euclidean distance
                if m[0].distance / m[1].distance <= args.ratio:
                    desc[i] = real_descriptors[e]
                    labels_out[i,0] = m[0].trainIdx
                    labels_real[i,0] = labels[k]
                    i += 1
                    if i >= args.max_descriptors[0]: 
                        break
            if i >= args.max_descriptors[0]: 
                break
            progress.update(k+1)
        progress.finish()
        print('have {} descriptors so far:'.format(i))
        if no_new:
            break
    if no_new:
        desc = desc[:i] # reduce descriptor
        labels_out = labels_out[:i]
        labels_real = labels_real[:i]
    else:
        assert(i == desc.shape[0]) 

    with open(os.path.join(args.outputfolder, 'samples.ocvmb'),
          'wb') as f:
        f.write(struct.pack('=ciii', 'f', 
                            desc.shape[0], desc.shape[1], 1))
        desc.tofile(f)
    
    with open(os.path.join(args.outputfolder, 'labels.ocvmb'),
          'wb') as f:
        f.write(struct.pack('=ciii', 'f', 
                            labels_out.shape[0], labels_out.shape[1], 1))
        labels_out.tofile(f)
        
    with open(os.path.join(args.outputfolder, 'labels_real.ocvmb'),
          'wb') as f:
        f.write(struct.pack('=ciii', 'f', 
                            labels_real.shape[0], labels_real.shape[1], 1))
        labels_real.tofile(f)

    with open(os.path.join(args.outputfolder, 'cmd.txt'), 'w') as f:
        f.write('{}\n'.format(' '.join(sys.argv)))
