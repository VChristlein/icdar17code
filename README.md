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
