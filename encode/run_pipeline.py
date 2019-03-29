import shutil
import os
import sys
import argparse
import ConfigParser
import ast
import numpy as np
import time
import copy
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import OneClassSVM
from sklearn import preprocessing
import progressbar

import puhma_common as pc
import preprocess
import clustering
import ubm_adaption
import evaluate
import classification
import exemplar_cls_quwi
import train_lcs
import train_tv_space
import exemplar_cls

def parse_skip(s):
    if s == '':
        return
    s = s.strip(' \t\n\r') # probably argparse does that already
    l = s.split(' ')
    return l

# TODO: parser not needed anymore -> remove
def updateArgs(prefix, prefix_dict, prefix_cmdline, args, parser):
    oldargs = copy.deepcopy(args)
    found = False
    # step 1: set new defaults from the prefix_dict which stem from
    # the config file
    if prefix in prefix_dict:
        print 'Add new args:', prefix_dict[prefix]
        vars(args).update(prefix_dict[prefix])

    # step 2: parse remaining arguments
    #         which stem from the command line
    if prefix in prefix_cmdline:
        # let's make a new parser
        update_parser = argparse.ArgumentParser()
        if prefix == 'p' or prefix =='p_fit'\
           or prefix == 'p_enc_fit' or prefix == 'p_enc':
            update_parser = preprocess.addArguments(update_parser)
        elif prefix == 'cl':
            update_parser = clustering.addArguments(update_parser)
        elif prefix == 'lcs':
            update_parser = train_lcs.addArguments(update_parser)
        elif prefix == 'tv':
            update_parser = train_tv_space.addArguments(update_parser)
        else: 
            raise ValueError('prefix >{}< unknown'.format(prefix))
        #parser = pc.commonArguments(parser)
        #parser = ubm_adaption.addArguments(parser)
        #parser = evaluate.parserArguments(parser)

        print 'Add new args from cmdline:', prefix_cmdline[prefix]
        newargs = update_parser.parse_args(prefix_cmdline[prefix])
        newargs = vars(newargs)
        # now we have to drop all arguments that were not explecitely given
        # ... dude how that sucks...
        all_keys = newargs.keys()
        for k in all_keys:
            tmp = '--' + k
            if tmp not in prefix_cmdline[prefix]:
                del newargs[k]
        print 'added newargs:', newargs
        vars(args).update(newargs)

    return args, oldargs

def setArg(args, member, to, verbose=True):
    vars(args)[member] = to
    if verbose:
        print 'set args.{} to {}'.format(member, to)
    return args

def copyConfig(outputfolder, config):
    cfg_filename = os.path.join(outputfolder, os.path.basename(config))
    cnt = 1
    # make sure file won't get overwritten if it exists
    old_filename = None
    while True:
        if not os.path.exists(cfg_filename):
            break
        else:
            old_filename = cfg_filename
        root,ext = os.path.splitext(os.path.basename(config))
        cfg_filename = os.path.join(outputfolder, root + str(cnt) + ext)
        cnt += 1
    # only copy if neccessary, i.e. files differ
    from hashlib import md5
    if old_filename:
        cfg_md5 = md5(open(config, 'rb').read()).digest()
        cfg_old_md5 = md5(open(old_filename, 'rb').read()).digest
        if cfg_md5 != cfg_old_md5:
            shutil.copyfile(config, cfg_filename)
    else:
        shutil.copyfile(config, cfg_filename)


def pipelineArguments(parser):
    parser.add_argument('--identifier',
                        help='name the experiment, if empty outputfoldersname'
                        ' is taken')
    parser.add_argument('--l_exp',
                        help='label of exp file')
    parser.add_argument('--l_ben',
                        help='label of ben file')
    parser.add_argument('--l_val',
                        help='label of validation file')
    parser.add_argument('--train-sample-weights',
                        help='file name of the sample weights for l_exp')
    parser.add_argument('--test-sample-weights',
                        help='file name of the sample weights for l_ben')
    parser.add_argument('--in_exp',
                        help='input of exp images')
    parser.add_argument('--in_ben',
                        help='input of ben images')
    parser.add_argument('--in_val',
                        help='input of validation images')
    parser.add_argument('--maskfolder_exp',
                        help='maskinput of exp images')
    parser.add_argument('--maskfolder_ben',
                        help='maskinput of ben images')
    parser.add_argument('--mask_enc', action='store_true',
                        help='use mask only for the encoding step')
    parser.add_argument('--post_exp',
                        help='posteriors dir for train')
    parser.add_argument('--post_ben',
                        help='posteriors dir for test')

    parser.add_argument('--suffix_ben',
                        help='if suffix of benchmark is different from'
                        ' suffix of exp set')
    parser.add_argument('--suffix_val',
                        help='if suffix of val')
    parser.add_argument('--run', choices=('prep_exp_fit', 'prep_exp', 'prep_ben',
                                          'cluster', 'train_lcs', 'train_tv',
                                          'enc_ben',
                                          'enc_exp', 'ex_cls',
                                          'prep_enc_exp_fit', 'prep_enc_exp',
                                          'prep_enc_ben', 'classify',
                                          'eval_prep_enc_ben',
                                          'eval_prep_enc_exp'),
                        nargs='+',
#                        default=['prep_exp','prep_ben',
#                                 'cluster', 'enc_ben', 'enc_exp'],
                        help='which classes to run')
    # TODO
    parser.add_argument('--cluster_times', type=int,
                        help='run clustering so often')
    parser.add_argument('--start_clusteridx', type=int, default=0,
                        help='dont start from zero if you already have computed'
                        ' some clusters')
    parser.add_argument('--individual_prep', '--prep_individual', action='store_true',
                        help='individual preprocessing of the encodings if'
                        ' multiple runs - instead of joint eval')
    parser.add_argument('--load', default=[],
                        type=parse_skip,
                        help='load stuff - similar to skip but for external'
                        ' folders')
#    parser.add_argument('--load2', default=[],
#                        type=parse_skip,
#                        help='load stuff - similar to skip but for external'
#                        ' folders')
    parser.add_argument('--skip', default=[],
                         type=parse_skip,
                        #nargs='*', default=[],
                        #choices=('prep_exp_fit', 'prep_exp', 'prep_ben',
                        #         'cluster', 'enc_ben',
                        #         'enc_exp', 'ex_cls',
                        #         'prep_enc_exp_fit', 'prep_enc_exp',
                        #         'prep_enc_ben'),
                        help='skip running of this class')
    parser.add_argument('--rm_run', '--ignore', default=[], type=parse_skip,
                        help='remove things in the run list, comes handy if you'
                        ' evaluate many configs')
    parser.add_argument('--add_run', default=[], type=parse_skip,
                        help='add things to the run list, comes handy if you'
                        ' evaluate many configs')
    parser.add_argument('--load_folder',
                        help='use this folder as base to load the skipped '
                        'stuff')
#    parser.add_argument('--load_folder2',
#                        help='use this folder as base to load the skipped '
#                        'stuff')
    parser.add_argument('--not_enc_args', action='store_true',
                        help='do not encode arguments as subfolders of the'
                        'output folder')
    # need to have this due to classification subparser
    parser.add_argument('-', dest='__dummy',
                        action='store_true', help=argparse.SUPPRESS)

    # TODO: move this to exemplar_cls.py or exemplar_cls_quwi.py
    parser.add_argument('--resampling', type=int,
                        help='resampling strategy for exemplar-cls w. SGD')
    parser.add_argument('--feature_encoder', type=int, default=0,
                        help='use exemplar-cls as feature encoder instead of'
                        ' classifier')
    parser.add_argument('--multi_pos', action='store_true', 
                        help='use multiple positives for one exemplar')
    # FIXME: do we still need it? does it work?
    parser.add_argument('--resume', action='store_true', help='help resumes'
                        ' aggregation FIXME')
    parser.add_argument('--not_independent', action='store_true',
                        help='are the train and test data not indpendent')
    parser.add_argument('--from_disk', action='store_true',
                        help='loads ecls from disk for prediction')
    parser.add_argument('--load_ex_fe', action='store_true',
                        help='load e-ecls features from disk')
    parser.add_argument('--n_cls', type=int,
                        help='compute this number of random e-cls')
    parser.add_argument('--dual_ex', action='store_true',
                        help='compute ecls scores')
    parser.add_argument('--attr_ex', action='store_true',
                        help='use background scores as attr')
    parser.add_argument('--multi_ex', action='store_true',
                        help='compute multi e-cls scores')
    parser.add_argument('--avg_neg', action='store_true',
                        help='average negative descriptors for E-SVMs')
    parser.add_argument('--nowait', action='store_true',
                        help='do not wait before running the pipeline')

    return parser

if __name__ == '__main__':
    conf_parser = argparse.ArgumentParser('run pipeline',  add_help=False)
    conf_parser.add_argument('-c', '--config',
                             help='Specify config file', metavar="FILE")
    args, remaining_args = conf_parser.parse_known_args()

	# let's restrict our resources
	# FIXME: doesn't do anything O_o (maybe due to old python2 version of f**king ubuntu)
#    import resource
#    #rsrc = resource.RLIMIT_DATA
#    #rsrc = resource.RLIMIT_VMEM
#    rsrc = resource.RLIMIT_STACK
#    soft, hard = resource.getrlimit(rsrc)
#    print 'Soft limit starts as  :', soft
#    
#    gb = 1024*1024*1024
#    resource.setrlimit(rsrc, (gb*30, gb*40)) #limit to 40gb 
#    
#    soft, hard = resource.getrlimit(rsrc)
#    print 'soft limit changed to :', soft 
#    print 'hard limit changed to :', hard 


    # parse arguments from the config file
    if args.config:
        print 'read config file', args.config
        assert(os.path.exists(args.config))
        config = ConfigParser.SafeConfigParser()
        config.read([args.config])

    #Don't supress add_help here so it will handle -h
    parser = argparse.ArgumentParser(
        # Inherit options from config_parser
        parents=[conf_parser],
        # print script description with -h/--help
        description=__doc__,
        # Don't mess with format of description
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # add arguments from other classes
    parser = pc.commonArguments(parser)
    parser = preprocess.addArguments(parser)
    parser = clustering.addArguments(parser)
    parser = ubm_adaption.addArguments(parser)
    parser = evaluate.parserArguments(parser)
    parser = train_lcs.addArguments(parser)
    parser = train_tv_space.addArguments(parser)
    # add pipeline args
    parser = pipelineArguments(parser)
    # and the classification args
    parser = classification.parseArguments(parser)

    prefix_dicts = {}
    if args.config:
        print 'set arguments from config file as default'
        args_dict = {}
        for k,v in config.items('arguments'):
            value = ast.literal_eval(v)
            if '.' in k:
                pref, rest = k.split('.')
                prefix_dicts.setdefault(pref,{})[rest] = value
                """
                prefix_dicts.setdefault(pref,[]).append('--{}'.format(rest))
                if isinstance(value, list) or isinstance(value, tuple):
                    for val in value:
                        prefix_dicts[pref].append('{}'.format(val))
                else:
                    prefix_dicts[pref].append('{}'.format(value))
                """
            else:
                args_dict[k] = value
        print 'config-params:',args_dict
        parser.set_defaults(**args_dict)

    print 'remaining_args:', remaining_args
    print 'parse remaining args acc to prefixes'
    # arguments with prefix
    prefix_cmdline = {}
    prefix = None
    new_rem_args = []
    for rem_arg in remaining_args:
        if rem_arg.startswith('--'):
            if '.' in rem_arg: 
                prefix, _ = rem_arg.split('.')
                #rest = rem_arg.replace(prefix + '.', '')
                rem_arg = rem_arg.replace(prefix + '.', '')
                prefix = prefix.replace('--','')
                prefix_cmdline.setdefault(prefix, []).append('--{}'.format(rem_arg))
            else:
                new_rem_args.append(rem_arg)
                prefix = None            
        else:
            if prefix is None:
                new_rem_args.append(rem_arg)
                continue
#                raise parser.error('Option {} cannot be parsed'.format(rem_arg))
#            prefix_cmdline.setdefault(prefix,{})[rest] = rem_arg
            prefix_cmdline[prefix].append(rem_arg)
    
    print 'prefix_cmdline:', prefix_cmdline
    print 'new remaining args:', new_rem_args

    # now parse all remining arguments
    args2, remaining_args = parser.parse_known_args(new_rem_args)

    # update the arguments
    vars(args2).update(vars(args))
    args = args2

    if args.seed:
        print '- set seed to {}'.format(args.seed)
        np.random.seed(args.seed)

    if not args.outputfolder:
        raise ValueError('no outputfolder given')

    print 'args.config:', args.config

    # remove or add run things 
    if args.rm_run:
        for to_rm in args.rm_run:
            if to_rm in args.run:
                args.run.remove(to_rm)
    if args.add_run:
        for to_add in args.add_run:
            args.run.append(to_add)

    # build outputfolder depending on number of arguments
    # and config-file, ignoring every option after --run
    if (args.config and len(sys.argv) > 3) or\
        (not args.config and len(sys.argv) > 1):
        print sys.argv
        s_l = []
        start = (3 if args.config else 2)
        skip = False
        for k in range(start,len(sys.argv),1):
            if sys.argv[k] == '-':
                continue
            if sys.argv[k] == '--skip' or sys.argv[k] == '--load_folder'\
               or sys.argv[k] == '--load':
                #or sys.argv[k] == '--load2' or\
#               sys.argv[k] == '--load_folder2':
                skip = True
                continue
            if skip:
                skip = False
                continue
            if sys.argv[k] == '--run' \
               or sys.argv[k] == '--cluster_times':
                break
            if sys.argv[k].startswith('--'):
                continue
            if ',' in sys.argv[k] or ':' in sys.argv[k]:
                s_l.append(sys.argv[k].replace(',','_').replace(':','_'))
            else:
                s_l.append(sys.argv[k])
        s = '_'.join(map(str,s_l))

        if s != '' and not args.not_enc_args:
            args.outputfolder = os.path.join(args.outputfolder, s)


    if not os.path.exists(args.outputfolder):
        pc.mkdir_p(args.outputfolder)
    else:
        if not args.nowait:
            print 'WARNING: outputfolder already exists! Wait 3 sec...'
            time.sleep(3)

    if args.log:
        log = pc.Log(sys.argv, args.outputfolder, filename=args.logfile)

    # and copy ourself in the outputfolder, guarantees reproducability
    if args.config:
        copyConfig(args.outputfolder, args.config)

    # create temporary folders for each run-command
    print 'global outputfolder:', args.outputfolder
    folders = {}
    for d in args.run:
        # use skipped stuff from the skip_folder
        if d in args.load and args.load_folder:
            outputfolder = args.load_folder
#        elif d in args.load2 and args.load_folder2:
#            outputfolder = args.load_folder2
        else:
            outputfolder = args.outputfolder
        # only for clustering and the encoding step
        if args.cluster_times and ('cluster' in d or d.startswith('enc') or\
                                   'train_lcs' in d or 'train_tv' in d or\
                                   (args.individual_prep and\
                                    (d.startswith('prep_enc') or \
                                     d == 'prep_enc_exp_fit' or \
                                     d == 'eval_prep_enc_ben' or\
                                     d == 'eval_prep_enc_exp' or\
                                     d == 'classify' or\
                                     d == 'ex_cls'))\
                                  ):
            all_f = []
            for i in range(args.cluster_times):
                folder = os.path.join(outputfolder, str(i), d)
                all_f.append(folder)
                pc.mkdir_p(folder)
            folders[d] = all_f
        else:
            folder = os.path.join(outputfolder, d)
            folders[d] = folder
            pc.mkdir_p(folder)

    # add every load-cmd to the skip
    #(it was only used to differentiate between local and external folders
    args.skip.extend(args.load)

    # save suffix
    if isinstance(args.suffix, basestring):
        setArg(args, 'suffix', [args.suffix])
    suf = args.suffix[0].split('.', 1)
    suf_exp = [ suf[0] if len(suf) > 0 else '' ]
    ext_exp = [ '.' + suf[1] if len(suf) > 1 else suf[0]]
    cur_in_exp = args.in_exp
    if args.suffix_ben:
        suf = args.suffix_ben.split('.', 1)
    suf_ben = [ suf[0] if len(suf) > 0 else '']
    ext_ben = [ '.' + suf[1] if len(suf) > 1 else suf[0]]
    cur_in_ben = args.in_ben
    stats = None

    if args.identifier:
        identifier = args.identifier
    else:
        identifier = os.path.basename(outputfolder).replace('_', '+')

    if 'prep_exp_fit' in args.run:
        log.log('-> prep_exp_fit')
        print '\nRUN preprocess fitting for experimentation set'
        args, oldargs = updateArgs('p_fit', prefix_dicts, prefix_cmdline, args, parser)
        setArg(args, 'inputfolder', cur_in_exp)
        setArg(args, 'outputfolder', folders['prep_exp_fit'])
        setArg(args, 'labelfile', args.l_exp)
        if 'prep_exp_fit' not in args.skip:
            preprocess.run(args)
        else: print('skip')
        args = oldargs
        setArg(args, 'load_dir', folders['prep_exp_fit'])
        log.log('<- prep_exp_fit')

    # preprocess exp
    if 'prep_exp' in args.run:
        log.log('-> prep_exp')
        print '\nRUN preprocess for experimentation set'
        args, oldargs = updateArgs('p', prefix_dicts, prefix_cmdline, args, parser)
        print 'args.load_dir:', args.load_dir
        setArg(args, 'inputfolder', cur_in_exp)
        setArg(args, 'outputfolder', folders['prep_exp'])
        setArg(args, 'labelfile', args.l_exp)
        setArg(args, 'suffix', suf_exp[0] + ext_exp[0])
        if 'prep_exp' not in args.skip:
            preprocess.run(args)
        else: print('skip')
        suf_exp[0] += '_pr'
        ext_exp[0] = '.pkl.gz'
        cur_in_exp = folders['prep_exp']
        args = oldargs
        log.log('<- prep_exp')

    if 'prep_ben' in args.run:
        log.log('-> prep_ben')
        print '\nRUN preprocess for benchmark set'
        args, oldargs = updateArgs('p', prefix_dicts, prefix_cmdline, args, parser)
        print 'args.load_dir:', args.load_dir
        setArg(args, 'inputfolder', cur_in_ben)
        setArg(args, 'outputfolder', folders['prep_ben'])
        setArg(args, 'labelfile', args.l_ben)
        setArg(args, 'suffix', suf_ben[0] + ext_ben[0])
        if 'prep_ben' not in args.skip:
            preprocess.run(args)
        else: print('skip')
        suf_ben[0] += '_pr'
        ext_ben[0] = '.pkl.gz'
        cur_in_ben = folders['prep_ben']
        args = oldargs
        log.log('<- prep_ben')

    ubm_path = None
    if 'cluster' in args.run:
        log.log('-> cluster')
        print '\nRUN clustering for experimentation set'
        args, oldargs = updateArgs('cl', prefix_dicts, prefix_cmdline, args, parser)
        setArg(args, 'inputfolder', cur_in_exp)
        if not args.mask_enc:
            setArg(args, 'maskfolder', args.maskfolder_exp)
        setArg(args, 'labelfile', args.l_exp)
        setArg(args, 'suffix', suf_exp[0] + ext_exp[0])
        setArg(args, 'posteriors_dir', args.post_exp)
        if args.cluster_times:
            all_ubm_paths = []
            for i in range(args.cluster_times):
                setArg(args, 'outputfolder', folders['cluster'][i])
                if 'cluster' not in args.skip and i >= args.start_clusteridx:
                    ubm_path = clustering.run(args)
                else:
                    ubm_path = os.path.join(folders['cluster'][i], args.method\
                                            + '.pkl.gz')
                    if not os.path.exists(ubm_path):
                        # check if user gave a path, then use that one                
                        if args.load_ubm is not None:
                            ubm_path = os.path.join(args.load_ubm,i,'cluster', args.method\
                                                    + '.pkl.gz')
                            if os.path.exists(ubm_path):
                                ubm_path = args.load_ubm
                            else:
                                raise ValueError('ubm path of load_ubm doesnt \
                                                 exist')

                # TODO: remove me - this is bullshit!
                #if args.seed:
                #    np.random.seed(args.seed + i + 1)
                all_ubm_paths.append(ubm_path)
            ubm_path = all_ubm_paths
        else:
            setArg(args, 'outputfolder', folders['cluster'])
            if 'cluster' not in args.skip:
                ubm_path = clustering.run(args)
            else:
                ubm_path = os.path.join(folders['cluster'], args.method\
                                            + '.pkl.gz')
                if not os.path.exists(ubm_path):
                    # check if user gave the path, then use that one                
                    if args.load_ubm is not None and\
                       os.path.exists(args.load_ubm):
                        ubm_path = args.load_ubm
        args = oldargs
        setArg(args, 'load_ubm', ubm_path)
        log.log('<- cluster')

    lcs_path = None
    if 'train_lcs' in args.run:
        log.log('-> train_lcs')
        if not args.load_ubm:
            raise ValueError('enc_ben option w.o. load_ubm doesnt work')
        ubm_path = args.load_ubm
        args, oldargs = updateArgs('lcs', prefix_dicts, prefix_cmdline, args, parser)

        # FIXME: something isn't reset here, e.g. skip train_lcs doesnt work if
        # we also skip enc_ben enc_exp ...
        setArg(args, 'maskfolder', args.maskfolder_ben)
        setArg(args, 'inputfolder', cur_in_exp)
        setArg(args, 'labelfile', args.l_exp)
        setArg(args, 'suffix', suf_exp[0] + ext_exp[0])

        if args.cluster_times:
            all_lcs_paths = []
            for i in range(args.cluster_times):
                setArg(args, 'load_ubm', ubm_path[i])
                setArg(args, 'outputfolder', folders['train_lcs'][i])
                lcs_path = os.path.join(folders['train_lcs'][i],
                                             'lcs.pkl.gz')
                if 'train_lcs' not in args.skip or not os.path.exists(lcs_path):
                    lcs_path = train_lcs.run(args)
                all_lcs_paths.append(lcs_path)
            lcs_path = all_lcs_paths
        else:
            setArg(args, 'load_ubm', ubm_path)
            setArg(args, 'outputfolder', folders['train_lcs'])
            if 'train_lcs' not in args.skip:
                lcs_path = train_lcs.run(args)
            else:
                lcs_path = os.path.join(folders['train_lcs'], 'lcs.pkl.gz')
        args = oldargs
        setArg(args, 'load_lcs', lcs_path)
        log.log('<- train_lcs')

#    print('-----')
#    print('Current Arguments:')
#    print(args)
#    print('-----')

    tv_path = None
    if 'train_tv' in args.run:
        log.log('-> train_tv')
        if not args.load_ubm:
            raise ValueError('enc_ben option w.o. load_ubm doesnt work')
        ubm_path = args.load_ubm

        # FIXME: something isn't reset here, e.g. skip train_tv doesnt work if
        # we also skip enc_ben enc_exp ...
        setArg(args, 'maskfolder', args.maskfolder_ben)
        setArg(args, 'inputfolder', cur_in_exp)
        setArg(args, 'labelfile', args.l_exp)
        setArg(args, 'suffix', suf_exp[0] + ext_exp[0])

        if args.cluster_times:
            all_tv_paths = []
            for i in range(args.cluster_times):
                setArg(args, 'load_ubm', ubm_path[i])
                setArg(args, 'outputfolder', folders['train_tv'][i])
                if 'train_tv' not in args.skip:
                    tv_path = train_tv_space.run(args)
                else:
                    tv_path = os.path.join(folders['train_tv'][i],
                                             'tv.pkl.gz')
                all_tv_paths.append(tv_path)
            tv_path = all_tv_paths
        else:
            setArg(args, 'load_ubm', ubm_path)
            setArg(args, 'outputfolder', folders['train_tv'])
            if 'train_tv' not in args.skip:
                tv_path = train_tv_space.run(args)
            else:
                tv_path = os.path.join(folders['train_tv'], 'tv.pkl.gz')
            setArg(args, 'load_tv_space', tv_path)
        log.log('<- train_tv')

    if 'enc_ben' in args.run:
        log.log('-> enc_ben')
        stats = None
        print '\nRUN encoding for benchmark set'
        if not args.load_ubm and args.encoding != 'sum' and 'gmp' not in args.encoding:
            raise ValueError('enc_ben option w.o. load_ubm doesnt work')
        if not ubm_path:
            if args.cluster_times:
                ubm_path = []
                for i in range(args.cluster_times):
                    path = os.path.join(args.load_ubm, str(i),
                                        'cluster', args.method +\
                                        '.pkl.gz')
                    if not os.path.exists(path):
                        raise ValueError('{} does not exist'.format(path))
                    ubm_path.append(path)
            else:
                ubm_path = args.load_ubm

        setArg(args, 'maskfolder', args.maskfolder_ben)
        setArg(args, 'inputfolder', cur_in_ben)
        setArg(args, 'labelfile', args.l_ben)
        setArg(args, 'suffix', suf_ben[0]+ ext_ben[0])
        setArg(args, 'posteriors_dir', args.post_ben)

        if args.cluster_times:
            for i in range(args.start_clusteridx, args.cluster_times):
                if lcs_path:
                    setArg(args, 'load_lcs', lcs_path[i])
                setArg(args, 'load_ubm', ubm_path[i])
                print 'load_ubm', ubm_path[i]
                if args.encoding == 'ivec' and 'enc_ben' not in args.skip:
                    setArg(args, 'load_tv_space', tv_path[i])
                tmp_outputfolder = folders['enc_ben'][i]
                if args.resume:
                    setArg(args, 'load_scores', tmp_outputfolder)
                setArg(args, 'outputfolder', tmp_outputfolder)
                if 'enc_ben' not in args.skip:
                    stats = ubm_adaption.run(args)
                if stats:
                    stats_fname = os.path.join(tmp_outputfolder, 'stats_ben.txt')
                    evaluate.write_stats(stats_fname, stats, identifier)

        else:
            if lcs_path:
                setArg(args, 'load_lcs', lcs_path)
            setArg(args, 'load_tv_space', tv_path)
            setArg(args, 'load_ubm', ubm_path)
            setArg(args, 'outputfolder', folders['enc_ben'])
            if args.resume:
                setArg(args, 'load_scores', folder['enc_ben'])
            if 'enc_ben' not in args.skip:
                stats = ubm_adaption.run(args)
            if stats:
                stats_fname = os.path.join(outputfolder, 'stats_ben.txt')
                evaluate.write_stats(stats_fname, stats, identifier)

#        suf_ben[0] += '_' + ( 'vlad' if args.encoding == 'pvlad' else\
        suf_ben[0] += '_' + args.encoding + '_' + args.update \
                    +  ('_' if len(args.normalize_enc) > 0 else '') +\
                    '_'.join(args.normalize_enc)
        ext_ben[0] = '.pkl.gz'
        cur_in_ben = folders['enc_ben']
        log.log('<- enc_ben')

    if 'enc_exp' in args.run:
        log.log('-> enc_exp')
        stats = None
        print '\nRUN encoding for experimental set'
        if not args.load_ubm and args.encoding != 'sum' and 'gmp' not in args.encoding:
            raise ValueError('enc_exp option w.o. load_ubm doesnt work')
        if not ubm_path:
            if args.cluster_times:
                ubm_path = []
                for i in range(args.cluster_times):
                    path = os.path.join(args.load_ubm, str(i),
                                        'cluster', args.method +\
                                        '.pkl.gz')
                    if not os.path.exists(path):
                        raise ValueError('{} does not exist'.format(path))
                    ubm_path.append(path)
            else:
                ubm_path = args.load_ubm
        setArg(args, 'maskfolder', args.maskfolder_exp)
        setArg(args, 'inputfolder', cur_in_exp)
        setArg(args, 'labelfile', args.l_exp)
        setArg(args, 'suffix', suf_exp[0] + ext_exp[0])
        setArg(args, 'posteriors_dir', args.post_exp)

        if args.cluster_times:
            for i in range(args.start_clusteridx, args.cluster_times):
                if lcs_path:
                    setArg(args, 'load_lcs', lcs_path[i])
                if args.encoding == 'ivec' and 'enc_exp' not in args.skip:
                    setArg(args, 'load_tv_space', tv_path[i])
                setArg(args, 'load_ubm', ubm_path[i])
                tmp_outputfolder = folders['enc_exp'][i]
                setArg(args, 'outputfolder', tmp_outputfolder)
                if args.resume:
                    setArg(args, 'load_scores', tmp_outputfolder)
                if 'enc_exp' not in args.skip:
                    stats = ubm_adaption.run(args)
                    if stats:
                        stats_fname = os.path.join(tmp_outputfolder, 'stats_exp.txt')
                        evaluate.write_stats(stats_fname, stats, identifier)
        else:
            if lcs_path:
                setArg(args, 'load_lcs', lcs_path)
            if args.encoding == 'ivec':
                setArg(args, 'load_tv_space', tv_path)
            setArg(args, 'outputfolder', folders['enc_exp'])
            if args.resume:
                setArg(args, 'load_scores', folder['enc_exp'])
            if 'enc_exp' not in args.skip:
                stats = ubm_adaption.run(args)
            if stats:
                stats_fname = os.path.join(outputfolder, 'stats_exp.txt')
                evaluate.write_stats(stats_fname, stats, identifier)
#        suf_exp[0] += '_' + ( 'vlad' if args.encoding == 'pvlad' else\
        suf_exp[0] += '_' +  args.encoding + '_' + args.update \
                    +  ('_' if len(args.normalize_enc) > 0 else '') +\
                    '_'.join(args.normalize_enc)
        ext_exp[0] = '.pkl.gz'
        cur_in_exp = folders['enc_exp']
        log.log('<- enc_exp')

    if 'prep_enc_exp_fit' in args.run:
        log.log('-> prep_enc_exp_fit')
        print '\nRUN Fitting of the encoded experimentation set'
        print 'remaining args', remaining_args
        args, oldargs = updateArgs('p_enc_fit', prefix_dicts, prefix_cmdline, args, parser)

        setArg(args, 'labelfile', args.l_exp)
        setArg(args, 'suffix', suf_exp[0] + ext_exp[0])

        if args.cluster_times and args.individual_prep:
            for i in range(args.cluster_times):
                setArg(args, 'inputfolder', folders['enc_exp'][i])
                setArg(args, 'outputfolder', folders['prep_enc_exp_fit'][i])
                if 'prep_enc_exp_fit' not in args.skip:
                    preprocess.run(args)

        else:
            setArg(args, 'outputfolder', folders['prep_enc_exp_fit'])
            if args.cluster_times:
                setArg(args, 'inputfolder',
                       [ os.path.commonprefix(cur_in_exp)] )
                setArg(args, 'inputfolders_suffix', 'enc_exp')
            else:
                setArg(args, 'inputfolder', cur_in_exp)
            if 'prep_enc_exp_fit' not in args.skip:
                preprocess.run(args)
        args = oldargs
        setArg(args, 'load_dir', folders['prep_enc_exp_fit'])
        log.log('<- prep_enc_exp_fit')

    # TODO:
#    if args.cluster_times and args.individual_prep:
#        if args.cluster_times:
    if 'prep_enc_exp' in args.run:
        log.log('-> prep_enc_exp')
        print '\nRUN preprocess for encoded experimentation set'
        args, oldargs = updateArgs('p_enc', prefix_dicts, prefix_cmdline, args, parser)
        #setArg(args, 'load_dir', folders['prep_enc_exp'])
        print 'args.load_dir:', args.load_dir
        setArg(args, 'labelfile', args.l_exp)
        setArg(args, 'suffix', suf_exp[0] + ext_exp[0])

        if args.cluster_times and args.individual_prep:
            for i in range(args.start_clusteridx, args.cluster_times):
                setArg(args, 'load_dir', folders['prep_enc_exp_fit'][i])
                setArg(args, 'inputfolder', folders['enc_exp'][i])
                setArg(args, 'outputfolder', folders['prep_enc_exp'][i])
                if 'prep_enc_exp' not in args.skip:
                    preprocess.run(args)

        else:
            setArg(args, 'outputfolder', folders['prep_enc_exp'])
            if args.cluster_times:
                setArg(args, 'inputfolder',
                       [ os.path.commonprefix(cur_in_exp)] )
                setArg(args, 'inputfolders_suffix', 'enc_exp')
            else:
                setArg(args, 'inputfolder', cur_in_exp)

            if 'prep_enc_exp' not in args.skip:
                preprocess.run(args)
        suf_exp[0] += '_pr'
        ext_exp[0] = '.pkl.gz'
        cur_in_exp = folders['prep_enc_exp']
        args = oldargs
        log.log('<- prep_enc_exp')

    if 'eval_prep_enc_exp' in args.run:
        log.log('-> eval_prep_enc_exp')
        print '\nRUN evaluate of prep exp'
        setArg(args, 'labelfile', args.l_exp)
        setArg(args, 'suffix', suf_exp[0] + ext_exp[0])

        if args.cluster_times and args.individual_prep:
            for i in range(args.start_clusteridx, args.cluster_times):
                setArg(args, 'inputfolder', cur_in_exp[i])
                setArg(args, 'outputfolder', folders['eval_prep_enc_exp'][i])
                if 'eval_prep_enc_exp' not in args.skip:
                    evaluate.run(args, identifier=identifier)

        else:
            setArg(args, 'inputfolder', cur_in_exp)
            setArg(args, 'outputfolder', folders['eval_prep_enc_exp'])
            if 'eval_prep_enc_exp' not in args.skip:
                evaluate.run(args, identifier=identifier)
        log.log('<- eval_prep_enc_exp')

    if 'prep_enc_ben' in args.run:
        log.log('-> prep_enc_ben')
        print '\nRUN preprocess for encoded benchmark set'
        args, oldargs = updateArgs('p_enc', prefix_dicts, prefix_cmdline, args, parser)
        setArg(args, 'suffix', suf_ben[0] + ext_ben[0])
        setArg(args, 'labelfile', args.l_ben)
        if args.cluster_times and args.individual_prep:
            for i in range(args.start_clusteridx, args.cluster_times):
                setArg(args, 'load_dir', folders['prep_enc_exp_fit'][i])
                print 'args.load_dir:', args.load_dir
                setArg(args, 'inputfolder', folders['enc_ben'][i])
                setArg(args, 'outputfolder', folders['prep_enc_ben'][i])
                if 'prep_enc_ben' not in args.skip:
                    preprocess.run(args)
        else:
            print 'args.load_dir:', args.load_dir
            setArg(args, 'outputfolder', folders['prep_enc_ben'])
            if args.cluster_times:
                setArg(args, 'inputfolder',
                       [ os.path.commonprefix(cur_in_ben)] )
                setArg(args, 'inputfolders_suffix', 'enc_ben')
            else:
                setArg(args, 'inputfolder', cur_in_ben)

            if 'prep_enc_ben' not in args.skip:
                preprocess.run(args)
        suf_ben[0] += '_pr'
        ext_ben[0] = '.pkl.gz'
        cur_in_ben = folders['prep_enc_ben']
        args = oldargs
        log.log('<- prep_enc_ben')

    if 'eval_prep_enc_ben' in args.run:
        log.log('-> eval_prep_enc_ben')
        print '\nRUN evaluate of prep ben'
        if args.cluster_times and args.individual_prep:
            for i in range(args.start_clusteridx, args.cluster_times):
                if args.not_independent:
                    setArg(args, 'inputfolders_probe', cur_in_ben[i])
                    setArg(args, 'labelfile_probe', args.l_ben)
                    setArg(args, 'inputfolder', cur_in_exp[i])
                    setArg(args, 'labelfile', args.l_exp)
                else:
                    setArg(args, 'inputfolder', cur_in_ben[i])
                    setArg(args, 'labelfile', args.l_ben)
                setArg(args, 'suffix', suf_ben[0] + ext_ben[0])
                setArg(args, 'outputfolder', folders['eval_prep_enc_ben'][i])
                if 'eval_prep_enc_ben' not in args.skip:
                    evaluate.run(args, identifier=identifier)
        else:
            if args.not_independent:
                setArg(args, 'inputfolders_probe', cur_in_ben)
                setArg(args, 'labelfile_probe', args.l_ben)
                setArg(args, 'inputfolder', cur_in_exp)
                setArg(args, 'labelfile', args.l_exp)
            else:
                setArg(args, 'inputfolder', cur_in_ben)
                setArg(args, 'labelfile', args.l_ben)
            setArg(args, 'suffix', suf_ben[0] + ext_ben[0])
            setArg(args, 'outputfolder', folders['eval_prep_enc_ben'])
            if 'eval_prep_enc_ben' not in args.skip:
                evaluate.run(args, identifier=identifier)
        log.log('<- eval_prep_enc_ben')

    # TODO: currently either classify or ex_cls
    # --> put classify after ex_cls, allow calibration
    if 'classify' in args.run:
        log.log('-> classify')
        print '\nRUN classifier'

        setArg(args, 'train_sample_weights', args.train_sample_weights)
        setArg(args, 'test_sample_weights', args.test_sample_weights)
        setArg(args, 'suffix', suf_exp[0] + ext_exp[0])
               #setArg(args, 'suffix', suf_ben[0] + ext_ben[0])

        def runClassify(in_ben, in_exp, out_classify):
            if args.use_ben_as_test:
                setArg(args, 'inputfolder', in_ben)
                setArg(args, 'labelfile', args.l_ben)
            elif args.use_exp_as_test:
                setArg(args, 'inputfolder', in_exp)
                setArg(args, 'labelfile', args.l_exp)

            else: # the common case
                setArg(args, 'inputfolder', in_exp)
                setArg(args, 'labelfile', args.l_exp)
                if args.not_independent:
                    setArg(args, 'ti', in_ben)
                    setArg(args, 'tl', args.l_ben)            
                else:
                    print 'independent train/test -> use inner grid_cv'
                    ' to get best classifier then set `use_?_as_test`'
                    ' option w. the best classifier'
            setArg(args, 'outputfolder', out_classify)

            classification.run(args)
       
        if 'classify' not in args.skip:
            if args.cluster_times and args.individual_prep:
                print 'run multiple classify'
                for i in range(args.start_clusteridx, args.cluster_times):
                    runClassify(cur_in_ben[i], cur_in_exp[i], folders['classify'][i])
            else:
                runClassify(cur_in_ben, cur_in_exp, folders['classify'])

    def excls(ben, exp, outputfolder, old_cls=None):
        stats = None
        setArg(args, 'outputfolder', outputfolder)

        inputfolders_suffix_ben = ''
        inputfolders_suffix_exp = ''
        if args.cluster_times and args.multi_pos > 0:
            inputfolders_suffix_ben = os.path.basename(ben[0])
            inputfolders_suffix_exp = os.path.basename(exp[0])
            ben = [os.path.commonprefix(ben)]
            exp = [os.path.commonprefix(exp)]
            print 'infolders:', ben, exp 
            print 'folder-suffixes:', inputfolders_suffix_ben, inputfolders_suffix_exp

        files, labels = pc.getFiles(ben, suf_ben[0]+ext_ben[0],
                                    labelfile=args.l_ben,
                                    exact=args.exact,
                                    inputfolders_suffix=inputfolders_suffix_ben)

        descr = pc.loadDescriptors(files)
        bg_files, bg_labels = pc.getFiles(exp, suf_exp[0]+ext_exp[0],
                                          labelfile=args.l_exp,
                                          exact=args.exact,
                                          inputfolders_suffix=inputfolders_suffix_exp)
        neg_descr = pc.loadDescriptors(bg_files)

        if args.avg_neg:
            classes = set(bg_labels)
            # this somehow should go more efficiently
            new_neg_d = {}
            for e,d in enumerate(neg_descr):
                new_neg_d.setdefault(bg_labels[e],[]).append(d)
                new_bg_labels = []
                new_neg = np.zeros( (len(classes), neg_descr.shape[1]),
                                   neg_descr.dtype)
                i = 0
                for k,v in new_neg_d.iteritems():
                    new_bg_labels.append(k)
                    desc_per_class = np.concatenate(v, axis=0)
                    desc_per_class = np.mean(new_neg, axis=0)
                    new_neg[i] = desc_per_class
                    i += 1

            bg_labels = new_bg_labels
            neg_descr = new_neg

        if descr.shape[1] != neg_descr.shape[1]:
            raise ValueError('descr.shape[1] != neg_descr.shape[1] -->'
                             ' forgot maybe <prep_enc_exp>?!')
        print 'desc and neg-desc shapes:', descr.shape, neg_descr.shape
        print '# labels and bg-labels:', len(labels), len(bg_labels)

        cls, grid = args.func(args)
        if grid and not args.load_ex_fe:
            # FIXME: better use an independent validation set (if avail)
            val_descr = None
            val_labels = None
            if args.in_val:
                val_files, val_labels = pc.getFiles(args.in_val, args.suffix_val,
                                    labelfile=args.l_val,
                                    exact=args.exact)
                val_descr = pc.loadDescriptors(val_files)

            if old_cls:
                cls = old_cls
            else:
                cls = exemplar_cls.crossVal(cls, grid, neg_descr, bg_labels,
                                            args.parallel, args.nprocs,
                                            n_descr_in_row = args.cluster_times if\
                                            args.multi_pos else -1,
                                            val_descr=val_descr, val_labels=val_labels)

        # use exemplar-classifier as feature encoder
        # Zpeda: "Exemplar SVMs as Visual Feature Encoder", CVPR15
        if args.feature_encoder > 0:
            desc_name = os.path.join(args.outputfolder, 'all_ex_fe.pkl.gz')
            if args.load_ex_fe:
                print 'load ex-fe descriptors'
                descr = pc.load(desc_name)
            else:
                descr, neg_descr = exemplar_cls_quwi.featureEnc(descr,
                                                            neg_descr, 
                                                            cls,
                                                            args.feature_encoder,
                                                            args.outputfolder,
                                                            bg_labels,
                                                            suffix='_ecls_fe.pkl.gz',
                                                            parallel=args.parallel,
                                                            nprocs=args.nprocs,
                                                            resampling=args.resampling,
                                                            grid=grid,
                                                            n_descr_in_row = args.cluster_times if\
                                                            args.multi_pos else -1)
                pc.dump(desc_name, descr)

            if args.not_independent:
                ret_matrix, stats = evaluate.classifyNN(descr, neg_descr, labels,
                                               bg_labels, distance=True,
                                               parallel=args.parallel,
                                               nprocs=args.nprocs)
            else:
                print 'evaluate now'
                ret_matrix, stats = evaluate.runNN(descr,
                                                   labels,
                                                   eval_method=args.eval_method,
                                                   distance=True,
                                                   parallel=args.parallel,
                                                   nprocs=args.nprocs)
            # write scores matrix
            pc.dump(os.path.join(args.outputfolder, 'scores_mat'), ret_matrix)
        else:
            if args.not_independent:
                print 'compute non-independent e-cls'
                ex_cls = exemplar_cls_quwi.computeExCls(neg_descr, cls,
                                                        len(neg_descr),
                                                        args.outputfolder,
                                                        bg_labels,
                                                        parallel=args.parallel,
                                                        nprocs=args.nprocs,
                                                        use_labels=True,
                                                        files=bg_files,
                                                        load=not args.overwrite,
                                                        suffix='_' +\
                                                            args.clsname +\
                                                           '.pkl.gz',
                                                       return_none=args.from_disk)
            else:
                ex_cls = exemplar_cls_quwi.computeIndependentExCls(descr,
                                                                   neg_descr,
                                                                   cls,
                                                                   outputfolder=args.outputfolder,
                                                                   parallel=args.parallel,
                                                                   nprocs=args.nprocs,
                                                                   resampling=args.resampling,
                                                                   files=files,
                                                                   load=not args.overwrite,
                                                                   suffix='_' +\
                                                                    args.clsname +\
                                                                    '.pkl.gz',
                                                                    return_none=args.from_disk,
                                                                   n_cls=args.n_cls)
                # FIXME: did this work? what does it do?
                if args.multi_ex:
                    print '> compute multi ex-cls'
                    # might get problems with similar names otherwise
                    multi_folder = args.outputfolder + '_multi'
                    pc.mkdir_p(dual_folder)
                    multi_ex_cls = exemplar_cls_quwi.computeExCls(neg_descr, cls,
                                                        len(neg_descr),
                                                        outputfolder=multi_folder,
                                                        labels=bg_labels,
                                                        parallel=args.parallel,
                                                        nprocs=args.nprocs,
                                                        use_labels=True,
                                                        files=bg_files,
                                                        load=not args.overwrite,
                                                        suffix='_' +\
                                                            args.clsname +\
                                                           '.pkl.gz',
                                                       return_none=args.from_disk)

            # we can use --overwrite False instead
#                if args.from_disk:
#                    scores_mat = exemplar_cls.predictLoadECLS(descr, args.outputfolder,
#                                                      bg_files if\
#                                                              args.not_independent\
#                                                              else files,
#                                                              suffix='_' +\
#                                                      args.clsname + '.pkl.gz',
#                                                      parallel=args.parallel, nprocs=args.nprocs)
#                else:

            print 'num ex-cls:', len(ex_cls)
            scores_mat = exemplar_cls_quwi.predict(files, ex_cls,
                                           parallel=args.parallel,
                                           nprocs=args.nprocs)

            # this does not equal to 'gallery adaptation' of Crosswhite et al.
            # but the spirit is similar
            if args.multi_ex:
                print 'num multi ex-cls:', len(multi_ex_cls)
                multi_scores_mat = exemplar_cls_quwi.predict(files, multi_ex_cls,
                                           parallel=args.parallel,
                                           nprocs=args.nprocs)
                print 'multi_scores_mat.shape:', multi_scores_mat.shape


            # this equals to 'probe adaptation' of Crosswhite et al.
            # TODO: think about it if that's correct
            if args.dual_ex:
                print '> compute dual-scores'
                upper = np.triu(scores_mat, 1)
                lower = np.tril(scores_mat, -1)
                scores_mat += upper.T
                scores_mat += lower.T

            stats = None
            if args.attr_ex:
                print '> use background scores as attributes'
                attr_scores_mat = exemplar_cls_quwi.predict(bg_files, ex_cls,
                                           parallel=args.parallel,
                                           nprocs=args.nprocs)
                print attr_scores_mat.shape

                _, stats = evaluate.runNN(attr_scores_mat.T, labels,
                                          distance=True,
                                          parallel=args.parallel,
                                          eval_method='euclidean',
                                          nprocs=args.nprocs)



            print 'scores_mat.shape:', scores_mat.shape
            if np.isnan(scores_mat).any():
                print 'WARNING have a nan in the scores_mat'
            if np.isinf(scores_mat).any():
                print 'WARNING have a inf in the scores_mat'

            log.log('<- ex_cls')

            log.log('-> eval')
            if stats is None and isinstance(cls, OneClassSVM):
                print '| compute stats from one-class svm'
                if args.not_independent:
                    stats = evaluate.computeStats('sum/max', scores_mat,
                                                  labels,  distance=True,
                                                  parallel=args.parallel,
                                                 labels_gallery=bg_labels,
                                                 eval_method=args.eval_method)
                else: #loo cross-val
                    if (scores_mat == np.finfo(scores_mat.dtype).max).any():
                        raise ValueError('there is already a float-maximum')
                    np.fill_diagonal(scores_mat, np.finfo(scores_mat.dtype).max)
                    stats = evaluate.computeStats('sum/max', scores_mat,
                                                  labels,
                                                  distance=True,
                                                  parallel=args.parallel,
                                                 eval_method=args.eval_method)
                    np.fill_diagonal(scores_mat,
                                     np.finfo(scores_mat.dtype).min)
                    stats = evaluate.computeStats('sum/max', scores_mat,
                                                  labels,
                                                  distance=False, parallel=args.parallel,
                                                 eval_method=args.eval_method)
            elif stats is None:
                if args.not_independent:
                    _, stats = evaluate.classifyNN(descr, neg_descr,
                                                   labels, bg_labels,
                                                   distance=False,
                                                   parallel=args.parallel,
                                                   nprocs=args.nprocs,
                                                   ret_matrix=scores_mat,
                                                  eval_method=args.eval_method)

#                    stats = evaluate.computeStats('sum/max', scores_mat,
#                                                  labels,
#                                                  distance=False,
#                                                  parallel=args.parallel,
#                                                  labels_gallery=bg_labels)
                else:
                    np.fill_diagonal(scores_mat, np.finfo(scores_mat.dtype).min)
                    stats = evaluate.computeStats('sum/max', scores_mat,
                                                  labels,
                                                  distance=False,
                                                  parallel=args.parallel,
                                                 nprocs=args.nprocs,
                                                 eval_method=args.eval_method)
            else:
                print 'stats already computed'

            # write scores matrix
            pc.dump(os.path.join(args.outputfolder, 'scores_mat'), scores_mat)

        # at this point everything already has to be pre/post-processed
#            stats = exemplar_cls_quwi.run(args,prep)

        stats_fname = os.path.join(args.outputfolder, 'stats_ecls.txt')
        evaluate.write_stats(stats_fname, stats, identifier)
        log.log('<- eval')
        return cls

    if 'ex_cls' in args.run:
        log.log('-> ex_cls')
        print '\nRUN exemplar cls'
   
        if args.cluster_times and args.individual_prep and not args.multi_pos:
            # we don't need to search the best SVM params in a grid for each
            # fold, so let's save it and use it
            old_cls = None 
            for i in range(args.start_clusteridx, args.cluster_times):
                if 'ex_cls' not in args.skip:
                    old_cls = excls(cur_in_ben[i], cur_in_exp[i], folders['ex_cls'][i],
                                    old_cls)
        else:
            if 'ex_cls' not in args.skip:
                cp = os.path.commonprefix(folders['ex_cls'])
                if cp == '':
                    excls(cur_in_ben, cur_in_exp, folders['ex_cls'])
                else:
                    excls(cur_in_ben, cur_in_exp,
                          os.path.join(args.outputfolder, 'ex_cls'))

# TODO: rm me 
#    if 'multi_ex_cls' in args.run and 'multi_ex_cls' not in args.skip:
#        assert(args.cluster_times > 1)
#        
#        files, labels = pc.getFiles(cur_in_ben, suf_ben[0]+ext_ben[0],
#                                    labelfile=args.l_ben, concat=True)
#        descr = pc.loadMultipleDescriptors(files)
#        print 'loaded multi-descriptors, shape=', descr.shape
#        print 'n labels:', len(labels)
#        bg_files, bg_labels = pc.getFiles(cur_in_exp, suf_exp[0]+ext_exp[0],
#                                          labelfile=args.l_exp, concat=True)
#        neg_descr = pc.loadMultipleDescriptors(bg_files)



    if args.log:
        log.dump()

