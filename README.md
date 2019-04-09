# icdar17code
Implementation of: 
Christlein, Vincent; Gropp, Martin; Fiel, Stefan; Maier, Andreas: "Unsupervised Feature Learning for Writer Identification and Writer Retrieval", ICDAR 2017

Please note that the parts are stripped from a larger code basis where parts are in C++, Python and Lua, so I hope it still is running. 
I will revise it as soon as possible, please drop me a line if you have questions.
The algorithm is actually also quite easy to implement, so everyone should be able to reproduce my results. 

Steps (no overall script currently...): 
- Extract SIFT decsriptors from the binarized ICDAR17 competition and at the same time extract 32x32 patches (eliminate doubles). 
	This code is in C++ and will be added soon, I will also upload the extracted patches and descriptors. 
- run the pipeline (`run_pipeline.py`) just for clustering the SIFT descriptors (or use `clustering.py` directly) 
- run `cluster_index.py` to create the cluster-indices / apply ratio criterium etc.
- run `ocvmb2tt.lua` to convert them to torch tensors
- create labelTrain file and sample files which you need for the CNN training
- train the resnet with `loadwriters_resnet_fb.lua` (yeah the name is bad, I will rename it soon)
- use `extractMultipleFeatures_resnet_fb.lua` to extract the new resnet features 
- use `run_pipeline.py` to run the pipeline. 

The code is as is, no warranty, etc. Please cite our work if you use my code or make some effort in improving it...

### Example command:
for run_pipeline.py:
`run_pipeline.py -c vlad_enc_ssr.cfg # load config
  # labels for testing/training
  --l_ben icdar17_labels_test.txt  
  --l_exp icdar17_labels_train.txt 
  --in_exp your_exp/ # input folder for training set
  --in_ben your_ben/ # input folder for testing set
  --suffix _SIFT_patch_pr.pkl.gz # suffix of the feature files 
  --cluster_times 5 # run 5 clusterings
  --rm_run prep_exp prep_ben prep_exp_fit # remove from the run list, here prep_exp and prep_ben were already computed in a previous run
  --add_run ex_cls # we can also add stuff to the run command of the standard config, here: exemplar-classification 
  --nowait # remove automatic waiting in case of overwriting stuff...
  --outputfolder your_output_folder 
  --identifier cluster_icdar17_bin_sift_patch32_no_dups_5000clusters_vlad_enc_ssr 
  --not_enc_args # dont encode arguments as folderstructure
  --feature_encoder 1 # let's use our feature encoder but only 1 run 
  --grid lsvm # compute the best C parameter from training set`
