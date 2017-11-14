require 'torch'
require 'paths'
require 'torchx'
require 'xlua'
require 'math'

BinaryIO = require 'BinaryIO'

cmd = torch.CmdLine()
cmd:option('-i', '', 'Input Directory')
cmd:option('-l', '', 'label file')
cmd:option('-suffix', '', 'suffix')
cmd:option('-otrain', '', 'train output dir')
cmd:option('-otest', '', 'test output dir')
cmd:option('-prep', false, 'preprocess data, i.e. zero-mean, std=1')
cmd:option('-s', 400000, 'Number of patches in output files')
cmd:option('-prop', 0.7, 'train test proportion')
cmd:option('-nowrite', false, 'dont do anything')
cmd:option('-randin', 0, 'use that many random input sample per file, 0=all')
params = cmd:parse(arg)

all_samples = {}
all_labels = {}

c = 1

iFile = torch.DiskFile(params.l)
iFile:quiet()
inputFiles = {}
while not iFile:hasError() do
    local iline = iFile:readString("*l")
    if iline == '' then break end
	local i1,i2 = string.find(iline, " ")
	if not i1 then print('pattern not found') return end

	local wid = tonumber(iline:sub(i1))
	if wid > 255 then
		print('ERROR: wid > 255 currently not possible')
		return
	end
--	print(fname.." : "..wid)

	local fname = iline:sub(1,(i1-1))
	fname = paths.basename(fname, paths.extname(fname))
	local dataReader = BinaryIO.new(paths.concat(params.i, fname .. params.suffix), 'r')
	
	local data = dataReader:readTensor()
	dataReader:close()

	if params.randin > 0 then
		local shuffle = torch.randperm(data:size(1))
		local thsize = data:size()
		num = math.min(params.randin,data:size(1))
		thsize[1] = num
		local th = torch.ByteTensor(thsize)
		for j=1, num do
			th[j] = data[ shuffle[j] ]
		end
		data = th
	end

	table.insert(all_samples, data)
	-- careful: only labels <= 255 will work!
	table.insert(all_labels, torch.ByteTensor(data:size(1)):fill(wid))
end

print('')

all_samples = torch.concat(all_samples, 1)
all_labels = torch.concat(all_labels, 1)
-- make labels start from 1
all_labels:add(-torch.min(all_labels)+1)
print('number of all patches:'.. all_samples:size(1))

n_files = math.floor(all_samples:size(1) / params.s)
rest = all_samples:size(1) % params.s

-- let's use 'prop' percent for train
n_files_train = math.floor(n_files * params.prop)

-- make data zero mean and std = 1
if params.prep then
	n_samples_train = n_files_train * params.s
	print('all train-samples:'..n_samples_train)
	-- compute mean and std only from the training set
	-- but apply it to all samples
--	train_samples = all_samples[{{1,n_samples_train}}]
--	m = train_samples::mean(1)
--	all_samples:add(-m:expandAs(all_samples))
--	std = train_samples:std(1)
--	all_samples:div(std:expandAs(all_samples))
	-- the code above doesnt work due to ByteTensor,
	-- so now gradually compute mean and std
	mean = torch.FloatTensor(1, 32, 32):fill(0)
	var = torch.FloatTensor(1, 32, 32):fill(0)
	for x=1, n_samples_train do
		sample = all_samples[x]:float()
		delta = sample - mean
		mean = mean + delta / x
		var = var + torch.cmul(delta,(sample - mean))
	end    
	variance = var / n_samples_train
	std = variance:sqrt()
	-- save it
	mean_var = {}
	table.insert(mean_var, variance)
	table.insert(mean_var, std)
	prep = torch.concat(mean_var, 1)
	print(prep:size())
	torch.save(paths.concat(params.otrain, 'prep.tt'), prep)
end	

if params.nowrite then
	return
end

paths.mkdir(params.otrain)
if n_files_train ~= n_files then
	paths.mkdir(params.otest)
end

shuffle = torch.randperm(all_samples:size(1))
for i=1, n_files do
	xlua.progress(i, n_files)
	local thsize = all_samples:size()
	if i ~= n_files then
		num = params.s
	else
		num = params.s + rest
	end
	thsize[1] = num

	if params.prep then
		th = torch.FloatTensor(thsize)
		lth = torch.ByteTensor(num)
		for j=1, num do
			th[j] = all_samples[ shuffle[j+(i-1)*params.s] ]:float() - mean
			th[j]:cdiv(std:expandAs(th[j]))
			lth[j] = all_labels[ shuffle[j+(i-1)*params.s] ]
		end
		--print(mean:size())
		--print(th:size())
		-- no plan why this doesnt work
--		th:add(-mean:expandAs(th))
--		th:cdiv(std:expandAs(th))
	else
		th = torch.ByteTensor(thsize)
		lth = torch.ByteTensor(num)
		for j=1, num do
			th[j] = all_samples[ shuffle[j+(i-1)*params.s] ]
			lth[j] = all_labels[ shuffle[j+(i-1)*params.s] ]
		end
	end

	if i <= n_files_train then
		torch.save(paths.concat(params.otrain, i-1 ..'_patches.tt'), th)
		torch.save(paths.concat(params.otrain, i-1 ..'_labels.tt'), lth)
	else
		torch.save(paths.concat(params.otest, i-1 ..'_patches.tt'), th)
		torch.save(paths.concat(params.otest, i-1 ..'_labels.tt'), lth)
	end
end

