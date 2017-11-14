require 'torchx'
require 'torch'
require 'cunn'
require 'cudnn'
require 'paths'
require 'optim'
require 'nngraph'

binaryIO = {}
function binaryIO:readHeader()
    self.dtype = self.f:readChar()
    self.rows = self.f:readInt()
    self.cols = self.f:readInt()
    self.channels = self.f:readInt()
--    print(self.dtype..' '..self.rows..' '..self.cols)
end
function binaryIO:writeHeader(dchar, rows, cols, channels)
    self.dtype = string.byte(dchar)
    self.rows = rows
    self.cols = cols
    self.channels = channels
    self.f:writeChar(self.dtype)
    self.f:writeInt(rows)
    self.f:writeInt(cols)
    self.f:writeInt(channels)
end
function binaryIO:writeTensor(tensor)
    local dchar = string.char(self.dtype)
    if dchar == 'u' then
        self.f:writeByte(tensor:storage())
    elseif dchar == 'i' then
        self.f:writeInt(tensor:storage())
    elseif dchar == 'f' then
        self.f:writeFloat(tensor:storage())
    elseif dchar == 'd' then
        self.f:writeDouble(tensor:storage())
    end
end
function binaryIO:readTensor()
    local storage
    local tensor
    local dchar = string.char(self.dtype)
    local sqrtc = math.sqrt(self.cols)
    if dchar == 'u' then
        storage = self.f:readByte(self.rows*self.cols*self.channels)
        tensor = torch.ByteTensor(storage, 1, torch.LongStorage{self.rows, self.channels, sqrtc, sqrtc})
    elseif dchar == 'i' then
        storage = self.f:readInt(self.rows*self.cols*self.channels)
        tensor = torch.IntTensor(storage, 1, torch.LongStorage{self.rows, self.channels, sqrtc, sqrtc})
    elseif dchar == 'f' then
        storage = self.f:readFloat(self.rows*self.cols*self.channels)
        tensor = torch.FloatTensor(storage, 1, torch.LongStorage{self.rows, self.channels, sqrtc, sqrtc})
    elseif dchar == 'd' then
        storage = self.f:readDouble(self.rows*self.cols*self.channels)
        tensor = torch.DoubleTensor(storage, 1, torch.LongStorage{self.rows, self.channels, sqrtc, sqrtc})
    end
    return tensor
end
function binaryIO:open(fname, fmode)
    self.f = torch.DiskFile(fname, fmode)
    self.f:binary()
    if fmode == 'r' then
        self:readHeader()
    end
end
function binaryIO:close()
    self.f:close()
end

cmd = torch.CmdLine()
cmd:option('-input', '', 'input folder of files')
cmd:option('-model', '', 'single model file to load')
cmd:option('-o', '', 'Output folder for the feature extraction')
cmd:option('-cl', 0, 'cut these many layers')
cmd:option('-batchSize', 128, 'Batch size')
cmd:option('-label', '', 'label-file')
cmd:option('-suffix', '', 'suffix')
cmd:option('-test', false, 'test directly from labels given in the label file')
cmd:option('-testOnly', false, 'test only directly from labels given in the label file')
cmd:option('-isimg', false, 'the input files are actually images')
cmd:option('-invert', false, 'invert images')
cmd:option('-mask_folder', '', 'corresponding image masks, e.g. contour files')
cmd:option('-mask_suffix', '', 'corresponding image contour')
cmd:option('-cudamodel', false, 'convert model to cuda-tensor, i.e. use cuda()')
cmd:option('-average', false, 'compute avg of features')
cmd:option('-colornormalize', false, 'normalize mean / std')
cmd:option('-nooverwrite', false, 'dont overwrite files')
params = cmd:parse(arg)

if not params.testOnly then
	paths.mkdir(params.o)
end

cmd:log(paths.concat(params.o, 'log.log'), params)

-- get all input files
inputFiles = {}
labels_per_file = {}
maskFiles = {}
--if not params.test then
--	local iFile = torch.DiskFile(params.input)
--	iFile:quiet()
--	while not iFile:hasError() do
--		local iline = iFile:readString("*l")
--		if iline == '' then break end
--		table.insert(inputFiles, iline)
--	end
--else
local iFile = torch.DiskFile(params.label)
iFile:quiet()
while not iFile:hasError() do
	local iline = iFile:readString("*l")
	if iline == '' then break end
	local i1,i2 = string.find(iline, " ")
	if not i1 then print('pattern not found') return end

	local fname = iline:sub(1,(i1-1))
	local wid = tonumber(iline:sub(i1))
	if params.test then 
		table.insert(labels_per_file, wid) 
	end

	fname = paths.basename(fname, paths.extname(fname))
	local path = paths.concat(params.input, fname .. params.suffix)
	table.insert(inputFiles, path)
	if params.isimg and params.mask_folder ~= '' then
		local m_path = paths.concat(params.mask_folder, fname .. params.mask_suffix)
		table.insert(maskFiles, m_path)
	end

end	


if params.test then
	all_labels = torch.Tensor(labels_per_file)
	all_labels:add(-torch.min(all_labels)+1)
	n_classes = torch.max(all_labels)
	print('clases min max '..torch.min(all_labels)..' '..n_classes)
	confusion = optim.ConfusionMatrix(n_classes)
	local_confusion = optim.ConfusionMatrix(n_classes)
end

-- :cuda() seems to be needed for some modules using cudnn instead cunn.
if params.cudamodel then
	model = torch.load(params.model):cuda()
else 
	model = torch.load(params.model)
end

print('model has #layers: ' .. #model.modules)
print('model-last-layer-type: ' ..torch.type(model:get(#model.modules)))
--assert(torch.type(model:get(#model.modules)) == 'nn.Linear')
for i=#model.modules-params.cl+1,#model.modules do
	print('remove layer: '..i)
	model:remove(i)
end
print('model has now #layers: ' .. #model.modules)

-- remove dropout and batch-normalization
model:evaluate()
collectgarbage()
allnn = nil

-- The model was trained with this input normalization
local meanstd = {
   mean = {174.288},
   std  = {41.5995},
}
--if params.isimg then
--	require 'image'
--	t = require '../resnet_fb/datasets/transforms'
--end

print('start extracting features of #inputFiles ' .. #inputFiles)
mbsize = params.batchSize
for k=1,#inputFiles do
	xlua.progress(k, #inputFiles)
	if params.nooverwrite then	
		local outname = paths.basename(inputFiles[k], paths.extname(inputFiles[k])) .. '.ocvmb'
		outname = paths.concat(params.o, outname)
		if paths.filep(outname) == true then
			goto continue
		end
	end
	local tensor
--	print('process: '..inputFiles[k])
	if params.isimg then
		require 'image'
		local t = require '../resnet_fb/datasets/transforms'
        local img = image.load(inputFiles[k], 1, 'float')
        if params.mask_folder ~= '' then
			local mask = image.load(maskFiles[k], 1, 'float')
--			print(mask:size())
--			print(img:size())
--			local transform = t.Compose{
--			t.ColorNormalize(meanstd),
			local transform = t.MultiBinaryCropMask(32, params.invert)
--			}
			assert(mask ~= nil)
			tensor = transform(img, mask)
		else
			local transform = t.Compose{
--			t.ColorNormalize(meanstd),
				t.MultiBinaryCrop(32, params.invert)
			}
			tensor = transform(img)
		end

--		print(tensor:size())
--		tensor = torch.cat(tensor,tensor,2) -- <- guess that was for GAN (pix2pix)
--		print(tensor:size())
	else

		binaryIO:open(inputFiles[k], 'r')
	--	local tensor_cpu = binaryIO:readTensor()
		tensor = binaryIO:readTensor():float()
		binaryIO:close()
		
		if params.colornormalize then
			local t = require '../resnet_fb/datasets/transforms'
			for i=1, tensor:size(1) do
				local transform = t.Compose{
					t.ColorNormalize(meanstd)
				}
				tensor[i] = transform(tensor[i])
			end
		end
	end

	local nsamples = tensor:size(1)
	local n_chunks = math.ceil(nsamples / mbsize)
	local sample_chunks = tensor:chunk(n_chunks, 1)

	local all_features = {}
    for i=1, n_chunks do
		collectgarbage(); collectgarbage();
--		xlua.progress(i, n_chunks)
		local features = model:forward(sample_chunks[i]:cuda()):float()
		if i == 1 and k == 1 then
			print('features:size(): ')
			print(features:size())
		end	
		if features:size(1) > 0 then
--			if features:dim() ~= 1 then
				features = torch.reshape(features, features:size(1), features:nElement()/features:size(1))
				if i == 1 and k == 1 then
					print('features:size(): ')
					print(features:size())
				end	
--			end
			table.insert(all_features, features)
		end
	end
	
	local all_features = torch.concat(all_features, 1)

	if params.average == true then
		print('compute average')
		all_features = torch.mean(all_features, 1)		
	end

	if params.test then 
		-- compute local acc. per hand
		_, ind = all_features:float():sort(2, true)
		out = ind:select(2,1)

		-- see which equal the output
		local labels
		local nn
		if params.average then
			labels = all_labels[k]
			nn = out:eq(labels)
		else 
			labels = torch.FloatTensor(all_features:size(1),1):fill(all_labels[k])
			nn = out:eq(labels:long())
		end
		print("own local accuracy (1): ".. (nn:sum() / nn:size(1))*100 .. ' %')
		
--		print('size features ')
--		print(all_features:size())
--		print('size labels ')
--		print(labels:size())
		-- WHYYY do you fail :(

--		local_confusion:batchAdd(all_features, labels)
--		local_confusion:updateValids()
--		print("Confusion Local accuracy (2) : "..local_confusion.totalValid*100 ..' %')
--		local_confusion:zero()

--		confusion:batchAdd(all_features, labels)
--		confusion:updateValids()
--		print("Current total accuracy: "..confusion.totalValid*100 ..' %')
		
		if allnn == nil then
			allnn = nn
		else
			allnn = torch.cat(allnn, nn, 1)
		end
		print("Current total accuracy: ".. (allnn:sum() / allnn:size(1))*100 ..' %')
	end


	-- write it out
	if not params.testOnly then
		outname = paths.basename(inputFiles[k], paths.extname(inputFiles[k])) .. '.ocvmb'
		binaryIO:open(paths.concat(params.o, outname), 'w')
		if i == 1 then
			print(all_features:size())
		end
		binaryIO:writeHeader('f', all_features:size(1), all_features:size(2), 1)
		binaryIO:writeTensor(all_features)
		binaryIO:close()
	end
	::continue::
end

if params.test then 
	print("Total accuracy: "..confusion.totalValid*100 ..'%')
end
