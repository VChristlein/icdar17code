require 'torch'
require 'paths'
require 'torchx'
require 'xlua'
require 'math'

local BinaryIO = require 'BinaryIO'

local cmd = torch.CmdLine()
cmd:option('-inf', '', 'input descriptor file')
cmd:option('-inl', '', 'input label file')
cmd:option('-ntest', '', 'number of test samples')
cmd:option('-of', '', 'output folder')
local params = cmd:parse(arg)
print(params)

torch.manualSeed(42)

paths.mkdir(params.of)

local dataReader = BinaryIO.new(params.inf, 'r')
local data = dataReader:readTensor()
dataReader:close()
print('data shape: ')
print(data:size())

dataReader = BinaryIO.new(params.inl, 'r')
local labels = dataReader:readTensor()
dataReader:close()
print('label old shape: ')
print(labels:size())
labels = torch.reshape(labels, labels:size(1), 1)
print('label new shape: ')
print(labels:size())
-- make labels start from 1
labels:add(-torch.min(labels)+1)

local ntest = tonumber(params.ntest)
local train = torch.FloatTensor(data:size(1) - ntest, data:size(2), data:size(3), data:size(4))
local test = torch.FloatTensor(ntest, data:size(2), data:size(3), data:size(4))
local tr_labels = torch.FloatTensor(data:size(1) - ntest, 1)
local te_labels = torch.FloatTensor(ntest, 1)

-- permutate data
shuffle = torch.randperm(data:size(1))
for i = 1,data:size(1) do
	if i <= ntest then
		test[i] = data[shuffle[i]]
		te_labels[i][1] = labels[shuffle[i]]
	else
		train[i-ntest] = data[shuffle[i]]
		tr_labels[i-ntest][1] = labels[shuffle[i]]
	end
end
print('tr / test shapes')
print(train:size())
print(test:size())
print('labels tr / test shapes')
print(tr_labels:size())
print(te_labels:size())
torch.save(paths.concat(params.of, 'patches_train.tt'), train)
torch.save(paths.concat(params.of, 'patches_test.tt'), test)
torch.save(paths.concat(params.of, 'labels_test.tt'), te_labels)
torch.save(paths.concat(params.of, 'labels_train.tt'), tr_labels)


--tr_labels = labels:narrow(1, 1, data:size(1)-params.ntest)
--print('tr_label new shape: ')
--print(tr_labels:size())
--te_labels = labels:narrow(1,data:size(1)-params.ntest,params.ntest)
--print('te_label new shape: ')
--print(te_labels:size())
