--require 'residual-layers'
require 'torch'
--require 'cutorch'
require 'nn'
--require 'cunn'
--require 'cudnn'
--require 'nngraph'
require 'optim'
--require 'image'
--require 'xlua'
require 'torchx'

--local nninit = require 'nninit'
require 'paths'

local models = require 'models/init'
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(2)
cutorch.manualSeedAll(2)


--binaryIO = {}
--function binaryIO:readHeader()
--    self.dtype = self.f:readChar()
--    self.rows = self.f:readInt()
--    self.cols = self.f:readInt()
--    self.channels = self.f:readInt()
----    print(self.dtype..' '..self.rows..' '..self.cols)
--end
--function binaryIO:writeHeader(dchar, rows, cols, channels)
--    self.dtype = string.byte(dchar)
--    self.rows = rows
--    self.cols = cols
--    self.channels = channels
--    self.f:writeChar(self.dtype)
--    self.f:writeInt(rows)
--    self.f:writeInt(cols)
--    self.f:writeInt(channels)
--end
--function binaryIO:writeTensor(tensor)
--    local dchar = string.char(self.dtype)
--    if dchar == 'u' then
--        self.f:writeByte(tensor:storage())
--    elseif dchar == 'i' then
--        self.f:writeInt(tensor:storage())
--    elseif dchar == 'f' then
--        self.f:writeFloat(tensor:storage())
--    elseif dchar == 'd' then
--        self.f:writeDouble(tensor:storage())
--    end
--end
--function binaryIO:readTensor()
--    local storage
--    local tensor
--    local dchar = string.char(self.dtype)
--    local sqrtc = math.sqrt(self.cols)
--    if dchar == 'u' then
--        storage = self.f:readByte(self.rows*self.cols*self.channels)
--        tensor = torch.ByteTensor(storage, 1, torch.LongStorage{self.rows, self.channels, sqrtc, sqrtc})
--    elseif dchar == 'i' then
--        storage = self.f:readInt(self.rows*self.cols*self.channels)
--        tensor = torch.IntTensor(storage, 1, torch.LongStorage{self.rows, self.channels, sqrtc, sqrtc})
--    elseif dchar == 'f' then
--        storage = self.f:readFloat(self.rows*self.cols*self.channels)
--        tensor = torch.FloatTensor(storage, 1, torch.LongStorage{self.rows, self.channels, sqrtc, sqrtc})
--    elseif dchar == 'd' then
--        storage = self.f:readDouble(self.rows*self.cols*self.channels)
--        tensor = torch.DoubleTensor(storage, 1, torch.LongStorage{self.rows, self.channels, sqrtc, sqrtc})
--    end
--    return tensor
--end
--function binaryIO:open(fname, fmode)
--    self.f = torch.DiskFile(fname, fmode)
--    self.f:binary()
--    if fmode == 'r' then
--        self:readHeader()
--    end
--end
--function binaryIO:close()
--    self.f:close()
--end
--
--function loadFile(fname)
--	if paths.extname(fname) == 'ocvmb' then
--		binaryIO:open(fname, 'r')
--		data = binaryIO:readTensor()
--		binaryIO:close()
--		return data 
--	else
--		return torch.load(fname)
--	end
--end

torch.setdefaulttensortype('torch.FloatTensor')

cmd = torch.CmdLine()
cmd:option('-exp', '', 'experimentfolder')
cmd:option('-p', false, 'Only print NN information')
cmd:option('-plot', false, 'Plot results after each epoch')
cmd:option('-labels', '', 'Label file')
cmd:option('-samples', '', 'sample file')
cmd:option('-tlabels', '', 'Test Label file')
cmd:option('-tsamples', '', 'Test sample file')
cmd:option('-testonly', false, 'Only run test')
cmd:option('-testN', -1, 'use only n samples for testing')
--cmd:option('-resume', '', 'Model file to resume training')
--cmd:option('-output', '', 'Output file')
cmd:option('-epochs', 5, 'Number of epochs')
--cmd:option('-log', 'log', 'Log folder')
-- Model and SGD Parameters
cmd:option('-Nsize', 2, 'Model as 8*n+2')
cmd:option('-batchSize', 128, 'Batch size')
--cmd:option('-dropout', false, 'Use dropout')
cmd:option('-learningRate', 0.01, 'Learning rate')
cmd:option('-learningRateDecay', 0, 'Learning rate decay')
cmd:option('-momentum', 0.9, 'Momentum for SGD')
cmd:option('-reset_momentum', 0, 'reset momentum after this number of epochs for SGD')
cmd:option('-dampening', -1, 'Dampening of momentum')
cmd:option('-weightDecay', 0.0001, 'Weight decay')
cmd:option('-nesterov', false, 'Use nesterov momentum')
cmd:option('-nClasses', 100, 'number of writers')
cmd:option('-backend', 'cudnn', 'Options: cudnn | cunn')
cmd:option('-cudnn', 'fastest', 'Options: fastest (default), deterministic')
cmd:option('-dataset', 'writer', 'writer')
cmd:option('-retrain', 'none', 'model to retrain')
cmd:option('-netType', 'preresnet', 'model to use [resnet | preresnet | lenet]')
cmd:option('-nGPU', 1, 'number of gpus')
cmd:option('-resetClassifier', false, 'reset the classifier, d.h. the last layer')
cmd:option('-createNewLabels', false, 'create new labels upon success of a patch')
cmd:option('-notest', false, 'dont test at all')
cmd:option('-augment', false, 'augment data by random translates')
cmd:option('-colornormalize', false, 'mean / std normalization')
cmd:option('-logSuffix', '', 'suffix for the log-file')
cmd:option('-noLRAdaptation', false, 'dont adapt the learning rate')
cmd:option('-optim', 'sgd', 'currently sgd and adadelta can be used')
cmd:option('-fac', 1, 'lr-multiplier fac')

--cmd:option('-outputSampleResults', 'sample_results.ocvmb', 'output the sample results (=confusion matrix')

params = cmd:parse(arg)

if params.nesterov then 
    params.dampening = 0
else
    if params.dampening < 0 then
        params.dampening = params.momentum
    end
end
if params.exp == '' then
	outdir = 'exp'..os.date("%Y%m%d%H%M")
	print('no experimentsfolder exp given, take: '..outdir)
else
	outdir = params.exp
end
paths.mkdir(outdir)
print('outdir:'..outdir)
cmd:log(paths.concat(outdir, 'log.log'), params)
train_logger = optim.Logger(paths.concat(outdir, 'train' .. params.logSuffix .. '.log'))
test_logger = optim.Logger(paths.concat(outdir, 'test' .. params.logSuffix .. '.log'))
--logger = optim.Logger(paths.concat(params.log, 'report.log'))

if not params.testonly then
    sFile = torch.DiskFile(params.samples)
    lFile = torch.DiskFile(params.labels)
    lFile:quiet()
    sFile:quiet()
    sampleFiles = {}
    labelFiles = {}
    while not sFile:hasError() and not lFile:hasError() do
        sline = sFile:readString("*l")
        lline = lFile:readString("*l")
        if sline == '' or lline == ''  then break end
        table.insert(sampleFiles, sline)
        table.insert(labelFiles, lline)
    end
	print('we got '..#labelFiles..' labelfiles')
	print('we got '..#sampleFiles..' samplefiles')
end

if not params.notest then
	sFile = torch.DiskFile(params.tsamples)
	lFile = torch.DiskFile(params.tlabels)
	lFile:quiet()
	sFile:quiet()
	sampleTestFiles = {}
	labelTestFiles = {}
	while not sFile:hasError() and not lFile:hasError() do
		sline = sFile:readString("*l")
		lline = lFile:readString("*l")
		if sline == '' or lline == ''  then break end
		table.insert(sampleTestFiles, sline)
		table.insert(labelTestFiles, lline)
	end
	print('we got '..#labelTestFiles..' testLabelfiles')
	print('we got '..#sampleTestFiles..' testSamplefiles')
end

local N = params.Nsize
local model, criterion = models.setup(params)

-- Classification criterion
--criterion = nn.ClassNLLCriterion()
--criterion = nn.MultiMarginCriterion()
-- if we use cross-entropy we dont need logsoftmax()
--criterion = nn.CrossEntropyCriterion()

mbsize = params.batchSize

classes = {}
for i=1,params.nClasses do
    table.insert(classes, i)
end
confusion = optim.ConfusionMatrix(classes)

-- Icdar17 color:
-- 174.288 41.5995
local meanstd = {
   mean = {174.288},
   std  = {41.5995},
   -- 0-1:
--	mean = {0.6834},
--	std = {0.16313},
}
-- Testing!
function test(n_test_samples,dolog)
	print('\nRunning on test set...')
	model:evaluate()
	if n_test_samples then 
		n_test_files = 1
	else
		n_test_files = #labelTestFiles
	end
	for f=1, n_test_files do
		collectgarbage(); collectgarbage();

		local samples = torch.load(sampleTestFiles[f]):float()
		local labels = torch.load(labelTestFiles[f]):float()

		if n_test_samples then
			samples = samples[{{1,n_test_samples}}]
			labels = labels[{{1,n_test_samples}}]
		end
		local nsamples = samples:size(1)

		if params.colornormalize then
			local t = require '../resnet_fb/datasets/transforms'
			for i=1, nsamples do
				local transform = t.Compose{
					t.ColorNormalize(meanstd)
				}
				samples[i] = transform(samples[i])
			end
		end

		--n_chunks = math.floor(nsamples / mbsize)
		local n_chunks = math.ceil(nsamples / mbsize)
		local sample_chunks = samples:chunk(n_chunks, 1)
		local label_chunks = labels:chunk(n_chunks, 1)

		-- write the output 
		local new_labels = {}
	    for i=1, n_chunks do
			xlua.progress(i, n_chunks)
			local inputs, targets

			local tmp_inputs = sample_chunks[i]
			local tmp_targets = label_chunks[i]
			
			-- create cudatensor if not already created
			inputs = inputs or (params.nGPU == 1
								and torch.CudaTensor()
								or cutorch.createCudaHostTensor())
			targets = targets or torch.CudaTensor()
			-- copy to cuda-tensor
			inputs:resize(tmp_inputs:size()):copy(tmp_inputs)
			targets:resize(tmp_targets:size()):copy(tmp_targets)

	        local output = model:forward(inputs)
	        confusion:batchAdd(output, targets)

			if params.createNewLabels then
				-- sort output descending
				_, ind = output:float():sort(2, true)
				out = ind:select(2,1)
				-- see which equal the output
				nn = out:eq(targets:long())
				table.insert(new_labels, nn)
			end			
	    end
		print(#new_labels)
		if params.createNewLabels then
			new_labels = torch.concat(new_labels, 1)
			ext = paths.extname(labelTestFiles[f])
			new_fname = paths.basename(labelTestFiles[f], ext)
			torch.save(paths.concat(params.exp, new_fname ..'_new.tt'), new_labels)
		end
--        confusion:updateValids()
--        print("\n Accuracy: "..confusion.totalValid*100 .."%")
	end -- end test files
	confusion:updateValids()
	print("Total accuracy on Test set: "..confusion.totalValid*100 ..'%')
	if dolog then
		test_logger:add{['test'] = confusion.totalValid * 100}
	end
	confusion:zero()
end

-- processed files
proc_samples = 0
if not params.resetClassifier then
	first, second, third, fourth, fith = true, true, true, true, true
end
fac = params.fac
if not params.testonly then
--	old_acc = 0
    for epoch=1,params.epochs do
		collectgarbage(); collectgarbage();
        print('Starting epoch ' .. epoch)

		-- reset momentum
		if params.reset_momentum > 0 and params.momentum > 0 and epoch > params.reset_momentum then
			params.momentum = 0
			params.nesterov = false
        end

		for f=1,#labelFiles do
			model:training()
            print('Using data from file ' .. sampleFiles[f])

            local samples = torch.load(sampleFiles[f]):float()
            local labels = torch.load(labelFiles[f]):float()
			collectgarbage(); collectgarbage();

			-- optimize via SGD using mini-batches of mbsize
            local nsamples = samples:size(1)
			if params.augment then
				local t = require '../resnet_fb/datasets/transforms'
				for i=1, nsamples do
					local transform = t.Compose{
						t.RandomCropOnepad(32,4)
					}
					samples[i] = transform(samples[i])
				end		
			end

			if params.colornormalize then
				local t = require '../resnet_fb/datasets/transforms'
				for i=1, nsamples do
					local transform = t.Compose{
						t.ColorNormalize(meanstd)
					}
					samples[i] = transform(samples[i])
				end
			end

			--local n_chunks = math.floor(nsamples / mbsize)
			local n_chunks = math.ceil(nsamples / mbsize)
            local shuffle = torch.randperm(n_chunks)
			local sample_chunks = samples:chunk(n_chunks, 1)
			local label_chunks = labels:chunk(n_chunks, 1)

            parameters, gradParameters = model:getParameters()
--            for t = 1, nsamples, mbsize do
            for t = 1, n_chunks do
				proc_samples = proc_samples + mbsize
				-- similar to the cifar dataset
				if not params.noLRAdaptation then 
					if fist and proc_samples < fac * 50000*80 then
						params.learningRate = 0.1
						first = false
					elseif second and proc_samples > fac * 50000*80 then
						print('reduce lr to 0.01')
						params.learningRate = 0.01
						second = false
					elseif third and proc_samples > fac * 50000*120 then
						print('reduce lr to 0.001')
						params.learningRate = 0.001
						third = false
--					elseif fourth and proc_samples > fac * 50000*160 then
--						print('reduce lr to 0.0001')
--						params.learningRate = 0.0001
--						fourth = false
--					elseif fith and proc_samples > fac * 50000*200 then
--						print('reduce lr to 0.00001')
--						params.learningRate = 0.00001
--						fith = false
					end
				end

                xlua.progress(t, n_chunks)
--				print('A')
				-- get random batch
				local tmp_inputs = sample_chunks[shuffle[t]]
				local tmp_targets = label_chunks[shuffle[t]]
--				print(tmp_inputs)
				-- create cudatensor if not already created
				inputs = inputs or (params.nGPU == 1
					      and torch.CudaTensor()
					      or cutorch.createCudaHostTensor())
--				inputs = inputs or torch.CudaTensor()					    
				targets = targets or torch.CudaTensor()
				-- copy to cuda-tensor
			    inputs:resize(tmp_inputs:size()):copy(tmp_inputs)
				targets:resize(tmp_targets:size()):copy(tmp_targets)

				-- twice?
				collectgarbage(); collectgarbage();

                local feval = function(x)
                    if x ~= parameters then
                        parameters:copy(x)
                    end

                    gradParameters:zero()

					local f = 0

					local output = model:forward(inputs)
					local err = criterion:forward(output, targets)
					f = f + err
					-- do we need that at all??
					model:zeroGradParameters()
					local df_do = criterion:backward(output, targets)
					model:backward(inputs, df_do)

					confusion:batchAdd(output, targets)

                    return f, gradParameters
                end -- end feval function

				if params.optim == 'sgd' then
	                _, loss = optim.sgd(feval, parameters, params)
				else
	                _, loss = optim.adadelta(feval, parameters, params)
				end
            end -- end minibatch function

            -- need to update accuracy values in confusion matrix now
            confusion:updateValids()
            print("\n Accuracy: "..confusion.totalValid*100 .."%")
			train_logger:add{['sam'] = proc_samples,
							['acc'] = confusion.totalValid * 100}
--            if params.plot then
--                logger:style{['% train mean class accuracy'] = '-'}
--                logger:plot()
--            end
			
--			if old_acc > confusion.totalValid then
--				params.learningRate = params.learningRate / 10.0
--			end
--			old_acc = confusion.totalValid

            confusion:zero()
			if not params.notest then 
				test(20000,true)
			end
		end -- files 

		torch.save(paths.concat(outdir,'temp_convnet_epoch_'..epoch..'.model'), model)
	end -- epochs

	print("Saving final model..")
	torch.save(paths.concat(outdir,'final.model'), model)
end

if not params.notest then 
	if params.testN > 0 then
		test(params.testN)
	else
		test(false, true)
	end
end

