require 'torch'
require 'nn'
require 'cutorch'
require 'image'
require 'cudnn'

cmd = torch.CmdLine()
cmd:option('-m', '', 'Model file')
cmd:option('-l', 1, 'layer number')
cmd:option('-f', 1, 'filter number')
params = cmd:parse(arg)

model = torch.load(params.m)

print(#model.modules)
weights = model.modules[params.l].weight
--local imgDisplay = image.toDisplayTensor{input=weights, padding=2} --, scaleeach=80}

numFilters = weights:size(1)
print(weights:size())
filtSize = math.sqrt(weights:size(2))
if filtSize == 1 then	
	local imgDisplay = image.toDisplayTensor{input=weights, nrow=numFilters, padding=1, scaleeach=80}
	image.save('filters_'..params.l..'_all.png', imgDisplay)
else
--filtSize = weights:size(3)
print('reshape to: '..filtSize)
for f=1,numFilters do
	print(weights[f]:size())	
--	if filtSize == 1 then
--	    local filter = weights[f]:reshape(numFilters, filtSize, filtSize)
	local imgDisplay = image.toDisplayTensor{input=weights[f], nrow=filtSize, padding=1, symmetric=true}
	image.save('filters_'..params.l..'_all_'..f..'.png', imgDisplay)
--    local filter = weights[f]:reshape(filtSize, filtSize)
--    local filter_cpu = filter:float()
--    filter_cpu = image.minmax{tensor=filter_cpu, symm=true}:reshape(1, filtSize, filtSize)
----    filter_cpu = image.scale(filter_cpu, 50,50)
--    image.save('filter_'..params.l..'_'..f..'.png', filter_cpu)
end
end

--maxval = math.max(math.abs(torch.max(filter_cpu)), math.abs(torch.min(filter_cpu)))
--filter_cpu:add(maxval)
--filter_cpu:mul(255/2*maxval)
--filter_cpu = filter_cpu:byte()
--filter_cpu = filter_cpu:reshape(1, filter_cpu:size(1), filter_cpu:size(2))



