import subprocess as sp
import socket
from ast import literal_eval
import gzip
import cv2
import cPickle
import os
import glob
import time
#import multiprocessing.dummy as multiprocessing
import multiprocessing
from multiprocessing.pool import ThreadPool
import numpy as np
import argparse
import sys
import progressbar
from skimage import morphology
from read_ocvmb import *
import shlex

verbose = False
if verbose:
    def verboseprint(*args):
        # Print each argument separately so caller doesn't need to
        # stuff everything to be printed into a single string
        for arg in args:
           print(arg),
        print()
else:
    verboseprint = lambda *a: None      # do-nothing function

def enable_nice_traceback():
    # also look for "ultraTB" in "IPython.CrashHandler"
    import IPython.core.ultratb as utb
    sys.excepthook = utb.VerboseTB(include_vars=0) #VerboseTB, ColorTB, AutoFormattedTB, FormattedTB

def commonArguments(parser):
    io_group = parser.add_argument_group('input output options')
    io_group.add_argument('-l', '--labelfile',
                          nargs='*',
                        help='label-file containing the images to load + labels')
    io_group.add_argument('--label_col', type=int, default=1,
                          help='which column to use as label from the'
                          ' label-file')
#    io_group.add_argument('--inputfolders',
#                        nargs='+',
#                        help='give some inputfolders or multiple ones')
    io_group.add_argument('--inputfolders_suffix',
                        default='',
                        help='if a suffix should be appended after to search for'
                        ' inputfolder/*/inputfolders_suffix')
    io_group.add_argument('--concat_axis',
                          default=1, type=int,
                          help='concatenation axis in case of multiple folders')
    io_group.add_argument('--min_descs',  default=0, type=int,\
                          help='minimum features per file')
    io_group.add_argument('-i', '--inputfolder',
                          '--inputfolders',
                          nargs='+',
                          help='the input folder of the images / features')
    io_group.add_argument('-s', '--suffix',
                          default=[''], nargs='*',
                          help='only chose those images with a specific suffix')
    io_group.add_argument('--exact', type=literal_eval, default=True,
                        help='between (stripped) label and suffix nothing is'
                          ' allowed, if set to false anything can be there')
    io_group.add_argument('-o', '--outputfolder', default='.',\
                        help='the output folder for the descriptors')
    io_group.add_argument('--group_word', action='store_true',
                          help='group by words, has to have a specific schema'
                          ' like puhma-snippets have, thus only this supported')
    io_group.add_argument('--maskfolder',
                          help='if given it will be searched for a mask in that'
                          'folder, the name must be the same as in the'
                          'inputfolder except of the suffix, an additional'
                          'masksuffix can be given')
    io_group.add_argument('--masksuffix', default='',
                          help='possible suffix a mask must have')
    io_group.add_argument('--overwrite', type=literal_eval, default=True,
                          help='overwrite output-file (default=True)')
    io_group.add_argument('--max_descriptors', nargs='*',
                          type=int,
                          help='load maximum descriptors')
    io_group.add_argument('--max_files', type=int, default=0,
                          help='use only this number of random files')
    io_group.add_argument('--posteriors_dir',
                           help='load the posteriors / cluster assignments from'
                           ' this dir')
    io_group.add_argument('--posteriors_suffix',
                           help='load the posteriors / cluster assignments from'
                           ' having this suffix')


    general = parser.add_argument_group('general options')
    # todo: add maybe verbosity levels
    general.add_argument('--debug', action='store_true',
                         help='debug option')
    general.add_argument('--log',
                         type=literal_eval, default=True,
                         help='log status')
    general.add_argument('--logfile', default='log.txt',
                         help='filename for the log-file')
    general.add_argument('--parallel', action='store_true',\
                        help='some parts are parallelized')
    general.add_argument('-n','--nprocs', type=int,
                         default=multiprocessing.cpu_count(),
                         help='number of parallel instances')
    general.add_argument('--lib',default='sklearn', choices=['sklearn', 'cv2',
                                                            'bob', 'bobhdf5'],\
                        help='use a different library (where possible), bob '
                        ' stuff is saved via bob, cv2 stuff is converted to'
                        ' sklearn')
    general.add_argument('--seed', type=int,
                        help='set seed for any randomness')
    general.add_argument('--no_write', action='store_true',
                         help='dont write temp files to harddrive')
    return parser

def getFilesGrouped(folder, pattern =''):
    """
    use this for puhma
    """
    all_files = glob.glob(os.path.join(folder, '*' + pattern))

    # group them according first two prefices, e.g.
    # calixt_ii__06748_11191011_officii_nostri_nos_ph_goe__6613__wort_ego_HARRIS_SIFT.csv
    # calixt_ii__06748_11191011_officii_nostri_nos_ph_goe__6614__wort_calixtus_HARRIS_SIFT.csv
    grouped = {}
    for f in all_files:
        splits = f.split('__', 2)
        base = splits[0] + '__' + splits[1]
        grouped.setdefault(base, []).append(f)

    grouped_files = []
    for base, files in grouped.iteritems():
        grouped_files.append(files)

    return grouped_files


def getFiles(folder, pattern, labelfile=None, exact=True,
             concat=False,
             inputfolders_suffix='',
             ret_label=None,
            label_col=1,
            max_files=0):
    assert(folder is not None)

    if ret_label != None:
        print ('WARNING: getFiles(): option `ret_label` is deprecated, if'
               ' ret_label==False everything might get wrong (2 instead of 1'
               ' argument')
    if isinstance(folder, list):
        if len(folder) == 1 and inputfolders_suffix == '':
            folder = folder[0]
    if pattern and isinstance(pattern, list):
        if len(pattern) == 1:
            pattern = pattern[0]
    if labelfile and isinstance(labelfile, list):
        if len(labelfile) == 1:
            labelfile = labelfile[0]

    if isinstance(folder, basestring):
        if not os.path.exists(folder):
            raise ValueError('folder {} doesnt exist'.format(folder))
        if not os.path.isdir(folder):
            raise ValueError('folder {} is not a directory'.format(folder))
            #return  [ folder ], None

    if isinstance(folder, list):
        return getMultipleFiles(folder,
                                inputfolders_suffix,
                                pattern,
                                labelfile=labelfile,
                                exact=exact,
                                concat=concat)
    if labelfile:
        labels = []
        with open(labelfile, 'r') as f:
            all_lines = f.readlines()
        all_files = []
        check = True
        for line in all_lines:
            # using shlex we also allow spaces in filenames when escaped w. ""
            splits = shlex.split(line)
            img_name = splits[0]
            if len(splits) > 1:
                class_id = splits[label_col]
#            except ValueError,IndexError:
            else:
                if check:
                    print ('WARNING: labelfile apparently doesnt contain a label, '
                          ' you can ignore this warning if you dont use the'
                          ' label, e.g. if you just want the'
                          ' files to be read in a certain order')
                    check = False
                img_name = line.strip(' \t\n\r')
                class_id = None
#            except:
#                raise

            # strip all known endings, note: os.path.splitext() doesnt work for
            # '.' in the filenames, so let's do it this way...
            for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb','.csv']:
                if img_name.endswith(p):
                    img_name = img_name.replace(p,'')
            if exact:
                #img_name = img_name.decode('utf-8').encode('utf-8')
                file_name = os.path.join(folder, img_name + pattern )
                #filename = unicode(file_name, 'utf-8')
                all_files.append(file_name)
            else:
                search_pattern = os.path.join(folder, img_name + '*' + pattern)
                files = glob.glob(search_pattern)
                if concat:
                    all_files.append(files)
                else:
                    all_files.extend(files)
            labels.append(class_id)
        # if one check failed we set the whole label-file to None
        if check == False:
            labels = None
    else:
        labels = None
        all_files = glob.glob(os.path.join(folder, '*' + pattern))

    if max_files > 0:
        ind = np.random.choice(len(all_files),
                               min(len(all_files), max_files),
                               replace=False)
        all_files = np.array(all_files)[ind]
        labels = np.array(labels)[ind]

    return all_files, labels

def getMaskFiles(files, suffix, maskfolder, masksuffix):
    if maskfolder == None:
        return None

    mask_found = 0
    masks = []
    for f in files:
        base = os.path.basename(f).replace(suffix,'')
        mask_name = os.path.join(maskfolder, base + masksuffix)
        if not os.path.exists(mask_name):
            masks.append(None)
        else:
            mask_found += 1
            masks.append(mask_name)

    if mask_found == 0:
        print ('WARNING: couldnt find any mask in maskfolder ({}) w.'
                'masksuffix({})'.format(maskfolder, masksuffix))
    else:
        print ('found {} maskfiles for {} files'.format(mask_found, len(files)))

    return masks

def loadDescriptors(files, reshape=False, max_descs=0,
                    max_descs_per_file=-1,
                    rand=True, ax=0, rm_zero_rows=False,
                    min_descs_per_file=0,
                    show_progress=True,
                    maskfiles=None,
                    return_index_list=False,
                    return_random_indices=False,
                    rand_indices=None,
                    concat_axis=1):
    if not max_descs:
        max_descs = -1
    if not isinstance(max_descs, int):
        print('WARNING max_descs != int -> try max_descs[0]:'
              ' {}'.format(max_descs[0]))
        # probably a tuple given which is not used atm
        max_descs = max_descs[0]

    #FIXME: loading multiple descriptors still
    # doesn't work correctly
    if len(files) == 0:
        print('WARNING: laodDescriptor() called with no files')
        return
    if isinstance(files, basestring):
        files = [files]
        if rand_indices is not None:
            rand_indices = [rand_indices]
    if maskfiles is not None and isinstance(maskfiles, basestring):
        maskfiles = [maskfiles]

#    print('should load multiple files:', files[0])
    if not isinstance(files[0], basestring):
    #if not isinstance(files, basestring):
        print('load multiple descriptors and concatenate them')
        assert(isinstance(files[0][0], basestring))
        files = zip(*files)
        # compute them from max_descs
        if max_descs_per_file == -1 and max_descs > 0:
            max_descs_per_file = max(int(max_descs/len(files[0])), 1)
            print 'max_descs_per_file', max_descs_per_file

        descs,\
                ind_list, \
                rand_ind_list = loadDescriptorsH(files[0],
                                                 max_descs=max_descs,
                                                 max_descs_per_file=max_descs_per_file,
                                                 rand=rand,
                                                 rm_zero_rows=rm_zero_rows,
                                                 min_descs_per_file=min_descs_per_file,
                                                 show_progress=show_progress,
                                                 return_index_list=True,
                                                 return_random_indices=True)
        print 'init descs.shape', descs.shape
        all_descs = [ descs ]
        for i in range(1,len(files)):
            assert(len(files[i]) == len(files[0]))
            descs, ind_list = loadDescriptorsH(files[i],
                                               max_descs=max_descs,
                                               rand=rand,
                                               max_descs_per_file=max_descs_per_file,
                                               rm_zero_rows=rm_zero_rows,
                                               min_descs_per_file=min_descs_per_file,
                                               show_progress=show_progress,
                                               return_index_list=True,
                                               rand_indices=rand_ind_list)
            print 'shape:', descs.shape
            all_descs.append(descs)
        if return_index_list:
            return np.concatenate(all_descs, axis=concat_axis), ind_list
        return np.concatenate(all_descs, axis=concat_axis)

    # else
    # compute them from max_descs
    if max_descs_per_file == -1 and max_descs > 0:
        max_descs_per_file = max(int(max_descs/len(files)), 1)

    return loadDescriptorsH(files,
                            reshape=reshape,
                            max_descs=max_descs,
                            max_descs_per_file=max_descs_per_file,
                            rand=rand, ax=ax, rm_zero_rows=rm_zero_rows,
                            min_descs_per_file=min_descs_per_file,
                            show_progress=show_progress,
                            maskfiles=maskfiles,
                            return_index_list=return_index_list,
                            return_random_indices=return_random_indices,
                            rand_indices=rand_indices)

def loadDescriptorsH(files, reshape=False, max_descs=0,
                     max_descs_per_file=-1,
                     rand=True, ax=0, rm_zero_rows=False,
                     min_descs_per_file=0,
                     show_progress=True,
                     maskfiles=None, return_index_list=False,
                     return_random_indices=False,
                     rand_indices=None):
    """load descriptors , pkl.gz or csv format
    """

    if len(files) <= 10 and show_progress:
        show_progress = False
    if show_progress:
        widgets = [progressbar.Percentage(), ' ', progressbar.Bar(), ' ',
               progressbar.ETA()]
        progress = progressbar.ProgressBar(widgets=widgets)
    else:
        def progress(x):
            return x

    if rand_indices is not None:
        assert(rand == True)

    index_list = [0]
    descriptors = []
    desc_length = 0
    random_indices = []
    for i in progress(range(len(files))):
        f = files[i]
        try:
            if f.endswith('pkl.gz'):
                with gzip.open(f, 'rb') as ff:
                    desc = cPickle.load(ff)
#                except IOError:
#                    print 'IOError at file {}'.format(f)
#                    raise
#                except:
#                    print
#                    raise
            elif f.endswith('ocvmb'):
                desc = readOcvmb(f)
            else:
                desc = np.loadtxt(f, delimiter=',',ndmin=2, dtype=np.float32)

            """
            # Note: costly operation which doesnt seem to bring anything
            if desc.dtype == np.uint8 and binary:
                desc /= np.max(desc) # 0/1
                assert(desc.shape[1] % 8 == 0)
                blocks = desc.shape[1] / 8
                new_desc = np.zeros( (desc.shape[0], blocks), dtype=np.uint8)
                for col in range(desc.shape[1]):
                    # note this actually computes it from left to right
                    # b = 7 - col % 8 would maybe more correct?
                    b = col % 8
                    new_desc[:,col/8] += desc[:,col] * 2**b
                desc = new_desc
            else:
            """
            if desc.dtype != np.float32 and desc.dtype != np.float64:
                #print ('WARNING: desc.dtype ({}) != np.float32 and !='
                #       ' np.float64 ->'
                #       ' convert it'.format(desc.dtype))
                desc = desc.astype(np.float32)
        except:
            print 'Error at file', f
            raise

        if not np.isfinite(desc).all():
            print >> sys.stderr, 'malformed descriptor in %s:' % f
            print >> sys.stderr, desc

            raise ValueError('descriptor contains non finite values')

        assert(np.isfinite(desc).all())

        # apply mask if available
        if maskfiles != None and maskfiles[i] != None:
            if maskfiles[i].endswith('pkl.gz') or maskfiles[i].endswith('ocvmb'):
                if maskfiles[i].endswith('ocvmb'):
                    mask = readOcvmb(maskfiles[i])
                else:
                    with gzip.open(maskfiles[i], 'rb') as ff:
                        mask = cPickle.load(ff)
                assert(len(mask) == len(desc))
                if mask.ndim > 1: # if we have more than one dimension
                    assert(mask.ndim == 2 and mask.shape[1] == 2)
                    mask = np.exp(mask[:,1])
            else:
                mask = cv2.imread(maskfiles[i], cv2.CV_LOAD_IMAGE_GRAYSCALE)
                if mask == None:
                    print 'WARNING couldnt read maskfile {}'.format(maskfiles[i])
                return
                mask = mask.ravel()
                assert(len(mask) == len(desc))
            #print mask.shape, desc.shape, np.count_nonzero(mask)
            n_before = desc.shape[0]
            desc = desc[ np.where(mask >= 0.5) ]
            print 'kept {} of {} descr.'.format(len(desc), n_before)

        if len(desc) == 0:
            print 'no descriptors of file {}?'.format(f)
            if maskfiles != None and maskfiles[i] != None:
                print ' and its maskfile {}'.format(maskfiles[i])
                print ('it has {} 0-entries '
                        'and {} !=0'.format(len(mask[mask==0]),
                                           len(mask[mask!=0])) )
            continue

        # remove zero-rows
        if rm_zero_rows:
            desc = desc[~np.all(desc == 0, axis=1)]
        # skip if too few descriptors per file
        if min_descs_per_file > 0 and len(desc) <= min_descs_per_file:
            continue
        # pick max_descs_per_file, either random or the first ones
        if max_descs_per_file > 0:
            if rand:
                #if len(desc) < max_descs_per_file:
                #    print "Oh Noes, error for file", f
                #    print "pixels in file", desc.shape
                #    print "maskpx in file", mask.shape
                if rand_indices:
                    indices = rand_indices[i]
                else:
                    indices = np.random.choice(len(desc),
                                           min(len(desc),
                                               max_descs_per_file),
                                           replace=False)
                desc = desc[ indices ]
                if return_random_indices:
                    random_indices.append(indices)
            else:
                desc = desc[:max_descs_per_file]
        # reshape the descriptor
        if reshape:
            desc = desc.reshape(1,-1)

        descriptors.append(desc)

        desc_length += len(desc)
        if return_index_list:
            index_list.append(desc_length)
        if max_descs > 0 and desc_length > max_descs:
            break

    if len(descriptors) == 0:
        if min_descs_per_file == 0:
            print 'couldnt load ', ' '.join(files)
        return None

    descriptors = np.concatenate(descriptors, axis=ax)

    if return_index_list:
        if return_random_indices:
            return descriptors, index_list, random_indices
        else:
            return descriptors, index_list
    if return_random_indices:
        return descriptors, random_indices

    return descriptors

def inFolders(in_inputfolders, in_inputfolders_suffix, in_patterns):
    inputfolders = None
    patterns = None
    if in_inputfolders and len(in_inputfolders) == 1 and \
       len(in_inputfolders_suffix) > 0:
        inputfolders =\
                    glob.glob(os.path.join(os.path.join(in_inputfolders[0],
                                              '*'),in_inputfolders_suffix))
        inputfolders = sorted(inputfolders)
    elif in_inputfolders:
        inputfolders = in_inputfolders
        if isinstance(inputfolders, basestring):
            inputfolders = [inputfolders]

    if isinstance(in_patterns, basestring):
        patterns = [in_patterns] * len(inputfolders)
    elif in_patterns and len(in_patterns) == 1:
        patterns = in_patterns * len(inputfolders)
    elif in_patterns:
        assert(len(inputfolders) == len(in_patterns))
    else:
        patterns = ['']*len(inputfolders)
    print 'folders, patterns:', inputfolders, patterns

    return inputfolders, patterns


def getMultipleFiles(folders,inputfolders_suffix,
                     pattern, labelfile=None, exact=True,
                     concat=False):
    inputfolders = None
    if len(folders) == 1 and \
       len(inputfolders_suffix) > 0:
        inputfolders =\
                    glob.glob(os.path.join(os.path.join(folders[0],'*'),
                                           inputfolders_suffix))
        inputfolders = sorted(inputfolders)
    else:
        inputfolders = folders

    if pattern and isinstance(pattern, basestring):
        pattern = [ pattern ] * len(inputfolders)

    if labelfile and isinstance(labelfile, basestring):
        labelfile = [ labelfile ] * len(inputfolders)

    if labelfile is None:
        labelfile = [ None ] * len(inputfolders)

    print 'inputfolders:', inputfolders
    print 'pattern:', pattern
    print 'labelfile:', labelfile

    all_files = []
    all_labels = []
    for f, pat, lbl in zip(inputfolders, pattern, labelfile):
        files, labels = getFiles(f, pat, lbl, exact)
        if concat:
            all_files.extend(files)
            all_labels.extend(labels)
        else:
            all_files.append(files)

    if not concat:
        all_files = zip(*all_files)
    else:
        labels = all_labels

    return all_files, labels

def loadMultipleDescriptors(files, concat_axis=1,
                            return_index_list=False):
    """
    assume files of the format: [('fileA_1', 'fileA_2'),
                                    ('fileB_1', 'fileB_2')]
    (of course each tuple can have more than two entries)
    Alternatively: a single tuple will work, too
    """
    if len(files) == 0:
        print 'WARNING: laodDescriptor() called with no files'
        return
    if isinstance(files, tuple):
        files = [files]
    if isinstance(files, basestring) or \
        not isinstance(files[0], tuple):
        raise ValueError('files must be a list of tuples, files[0]:'
                         ' {}'.format(files[0]))

    all_descriptors = []

    for f in files:
        descr, index_list = loadDescriptors(f,
                                           return_index_list=True,
                                           ax=concat_axis)
        all_descriptors.append(descr)
    descriptors = np.concatenate(all_descriptors, axis=0)
#    print 'new descriptors shape:', descriptors[0].shape

    if return_index_list:
        return descriptors, index_list

    return descriptors


def loadAllDescriptors(inputfolders,
                       inputfolders_suffix,
                       patterns,
                       labelfile,
                       concat_axis=1,
                       return_index_list=False):
    inputfolders, patterns = inFolders(inputfolders, inputfolders_suffix,
                                        patterns)
    print 'Inputfolders:\n' + '\n'.join(inputfolders)
    print 'Patterns:\n' + ', '.join(patterns)
    concat_enc = []
    all_files = []
    index_list = None
    labels = None
    for infolder, pattern in zip(inputfolders,patterns):
        files, labels = getFiles(infolder, pattern, labelfile=labelfile,
                                  exact=True)
        all_files.append(files)
        if len(files) != len(labels):
            raise ValueError('len(files) {} !='
                             'len(labels) {}'.format(len(files),
                                                 len(labels)))
        if return_index_list:
            descriptors, index_list = loadDescriptors(files,
                                                     return_index_list=True)
        else:
            descriptors = loadDescriptors(files)
        print 'descriptors shape:', descriptors.shape
        concat_enc.append(descriptors)

    if concat_axis == None or concat_axis == -1:
        descriptors = concat_enc
    else:
        descriptors = [ np.concatenate(concat_enc, axis=concat_axis) ]
        print 'new descriptors shape:', descriptors[0].shape

    if return_index_list:
        return descriptors, labels, all_files, index_list

    return descriptors, labels, all_files

def getMask(img_name, maskfolder, maskpattern='*.png'):
    if maskpattern == None:
        maskpattern = '*.png'
    if not maskfolder:
        return None
    base = os.path.splitext(os.path.basename(img_name))[0]
#    if '_' in base:
#        sep = True
#        base = base.split('_',1)[0] + '_'
    search_pattern = os.path.join(maskfolder, base + maskpattern)
    masks = glob.glob(search_pattern)
    if len(masks) == 0:
        print 'WARNING: no mask found for\n', img_name, '\nsearched for:', search_pattern
        return None
    if len(masks) > 1:
        print 'WARNING: more than one mask found for: ', img_name, 'take first one'
    return masks[0]

def mkdir_p(path, silent=False):
    try:
        os.makedirs(path)
    except OSError as exc:
        import errno
        if exc.errno == errno.EEXIST:
            pass
        else:
            raise
    else:
        if not silent:
            print 'created', path

def toBinary(mask):
    # test if not already binary
    if mask[mask==255].sum() != np.sum(mask):
        # maybe binary between 0,1?
        if mask[mask==1].sum() == mask.sum():
            mask *= 255
        else: # make it binary
           ret, mask = cv2.threshold(mask, 125, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    return mask

def createMask(img, mask_type=''):
    contours = None
    if mask_type == 'img':
        mask = toBinary(img)
        # invert result
        mask = np.bitwise_not(mask)
    elif mask_type == 'edge':
        # -1 --> Scharr
        #mask = cv2.Canny(img, 50, 200, apertureSize=-1, L2gradient=True)
        mask = cv2.Canny(img, 50, 200)
    elif mask_type == 'cc':
        # TODO: double check that
        # set to 1?
        mask[mask>1] = 1
        # invert ??
        mask = np.bitwise_not(img)
        contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_NONE)
        # change cnt to real masks
        #                for c in cnt:
#                    x = []
#                    y = []
#                    for p in c:
#                        x.append(p[0,0])
#                        y.append(p[0,1])
#                        #     cnt_p.append( [p[0,1], p[0,0]] )
#                    cnt_p = (np.array(y),np.array(x))
                    #img[cnt_p] = (255,0,0)
    elif mask_type == 'skeleton':
        mask = toBinary(img)
        mask = np.bitwise_not(mask)
        mask /= 255
        mask = morphology.skeletonize(mask)
    else:
        mask = np.ones( img.shape, dtype=np.uint8)
#    print mask_type

    return mask, contours


def spawn(f):
    def fun(q_in, q_out):
	while True:
	    i,x = q_in.get()
	    if i is None:
		break
	    q_out.put((i, f(x)))

    return fun

def parmap(f, iterable, nprocs=multiprocessing.cpu_count(),
           show_progress=False, size=None):
    """
    @param f
    function to be applied to the items in iterable
    @param iterable
    ...
    @param nprocs
    number of processes
    @param show_progress
    True <-> show a progress bar
    @param size
    number of items in iterable.
    If show_progress == True and size is None and iterable is not already a
    list, it is converted to a list first. This could be bad for generators!
    (If size is not needed right away for the progress bar, all input items
    are enqueued before reading the results from the output queue.)
    TLDR: If you know it, tell us the size of your iterable.
    """
    q_in = multiprocessing.Queue(1)
    q_out = multiprocessing.Queue()

    progress = None
    if show_progress:
	if not isinstance(iterable, list):
	    iterable = list(iterable)
	size = len(iterable)

	widgets = [ progressbar.Percentage(), ' ', progressbar.Bar(), ' ', progressbar.ETA() ]
	progress = progressbar.ProgressBar(widgets=widgets, maxval=size)

    proc = [ multiprocessing.Process(target=spawn(f), args=(q_in, q_out)) for _ in range(nprocs) ]

    for p in proc:
        p.daemon = True
        p.start()

    if progress is not None:
        progress.start()


    def enqueue():
        s = 0
        for i, x in enumerate(iterable):
            q_in.put((i,x))
            s += 1

        for _ in range(nprocs):
            q_in.put((None,None))

        return s

    pool = ThreadPool(processes=1)
    async_size = pool.apply_async(enqueue)

    if size is None:
        # this is the old behavior
        size = async_size.get()

    res = []
    progress_value = 0
    for _ in range(size):
        r = q_out.get()
        res.append(r)

        # we could: insert sorted, yield all results we have so far

        if progress is not None:
            progress_value += 1
            progress.update(progress_value)

    del pool
    for p in proc:
        p.join()

    if progress is not None:
        progress.finish()

    return [ x for _, x in sorted(res) ]

#def parmap(f, X, nprocs = multiprocessing.cpu_count()):
#    if nprocs is None: nprocs = multiprocessing.cpu_count()
#    q_in   = multiprocessing.Queue(1)
#    q_out  = multiprocessing.Queue()
#
#    proc = [multiprocessing.Process(target=spawn(f),args=(q_in,q_out)) for _ in range(nprocs)]
#    for p in proc:
#        p.daemon = True
#        p.start()
#
#    sent = [q_in.put((i,x)) for i,x in enumerate(X)]
#    [q_in.put((None,None)) for _ in range(nprocs)]
#    res = [q_out.get() for _ in range(len(sent))]
#
#    [p.join() for p in proc]
#
#    return [x for i,x in sorted(res)]

def dump(fname, obj, verbose=True, overwrite=True):
    filename = fname
    suffix = ''
    # FIXME suffix
    if not fname.endswith('.pkl.gz'):
        suffix = '.pkl.gz'
    if overwrite:
        filename += suffix
    else:
        ins = ''
        cnt = 0
        while os.path.exists(fname + str(ins) + suffix):
            ins = 1 + cnt
            cnt += 1
        filename = fname + str(ins) + suffix

    with gzip.open(filename, 'wb') as fOut:
        cPickle.dump(obj, fOut, -1)
    if verbose:
        print 'dumped', filename

def load(filename):
    if not filename.endswith('.pkl.gz'):
        filename += '.pkl.gz'
    with gzip.open(filename, 'rb') as f:
        ret = cPickle.load(f)
    return ret

class Log(object):
    def __init__(self,
                 args,
                 outputfolder,
                 filename='log.txt',
                 extra_cmd_file=True,
                 global_log=True):
        self.args = args
        self.start_time = time.time()
        self.outputfolder = outputfolder
        self.filename = filename
        self.msg = ''
        self.extra_cmd_file = extra_cmd_file
        self.global_log = global_log

        cmd_args = ' '.join(self.args)
        if self.extra_cmd_file:
            with open(os.path.join(self.outputfolder, 'cmd.txt'), 'a') as f:
                f.write(cmd_args+'\n')

        if self.global_log:
            fname = 'log_global_' + socket.gethostname() + '.txt'
            with open(fname, 'a') as f:
                f.write(cmd_args+'\n')

    def dump(self):
        duration = time.time() - self.start_time
        cmd_args = ' '.join(self.args)

        with open(os.path.join(self.outputfolder, self.filename), 'a') as f:

            f.write('\n---Command:---\n')
            f.write(cmd_args+'\n')

            if self.msg != '':
                f.write('---Message(s)---\n')
                f.write(self.msg)
                f.write('\n')

            f.write('Execution Start Time: {}\n'.format(time.strftime("%y/%m/%d , %H:%M:%S",
                                                                time.gmtime(self.start_time))))
            f.write('Duration:{:.0f}h {:.0f}m {:.0f}s\n'.format(duration / 3600,
                                                            (duration % 3600) / 60,
                                                            duration % 60))

            # Check whether the current working directory is an SVN working
            # directory. Run 'svn diff' only if this is the case to avoid
            # spurious warnings.
            svndir = os.path.join(os.getcwd(), '.svn')

            if os.path.isdir(svndir):
                f.write('SVN Revision: {}\n'.format(self.getRevision()))
                f.write('SVN diff: {}\n'.format(self.getSVNDiff()))

            f.write('---------------\n\n')

    def log(self, what):
        self.msg += '{} [{}]\n'.format(what, time.strftime("%H:%M:%S",
                                                           time.gmtime(time.time())) )

    def getSVNDiff(self):
        return sp.Popen(['svn', 'diff'], stdout=sp.PIPE).communicate()[0]

    def getRevision(self):
        return sp.check_output("svnversion")

def check(np_arr, what):
    if np.isnan(np_arr).any():
        raise ValueError('nan in {}!'.format(what))
    if np.isinf(np_arr).any():
        raise ValueError('inf in {}!'.format(what))
    if not np_arr.any():
        print 'WARNING: complete {}  0!'.format(what)

