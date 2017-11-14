--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local ELU = nn.ELU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
	local depth = opt.depth or nil
	local shortcutType = opt.shortcutType or 'B'
	local iChannels

	imgsize = 32
	nstates = {16, 256, 64}
	numWriters = 100
	filtsize = {7, 5}
	poolsize = {2, 3}

	-- nodes at after filter stages
	imgsizes = {imgsize}
	imgsizes[2] = (imgsizes[1]-filtsize[1]+1)/poolsize[1]
	imgsizes[3] = (imgsizes[2]-filtsize[2]+1)/poolsize[2]

	model = nn.Sequential()

    -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
    model:add(Convolution(1, nstates[1], filtsize[1], filtsize[1]))
    model:add(ReLU(true))
    model:add(Max(poolsize[1],poolsize[1],poolsize[1],poolsize[1]))

    -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
    model:add(Convolution(nstates[1], nstates[2], filtsize[2], filtsize[2]))
    model:add(ReLU(true))
    model:add(Max(poolsize[2],poolsize[2],poolsize[2],poolsize[2]))

    -- stage 3 : standard 2-layer neural network
    model:add(nn.View(nstates[2]*imgsizes[3]*imgsizes[3]))
    model:add(nn.Linear(nstates[2]*imgsizes[3]*imgsizes[3], nstates[3]))
    model:add(ReLU(true))

    --  model:add(nn.Dropout(0.5))
    -- model:add(nn.Linear(nstates[3], nstates[4]))
    -- model:add(nn.ReLU())

    -- classification stage
    model:add(nn.Linear(nstates[3], opt.nClasses))

	local function ConvInit(name)
		for k,v in pairs(model:findModules(name)) do
			local n = v.kW*v.kH*v.nOutputPlane
			v.weight:normal(0,math.sqrt(2/n))
			if cudnn.version >= 4000 then
				v.bias = nil
				v.gradBias = nil
			else
				v.bias:zero()
			end
		end
	end
	local function BNInit(name)
		for k,v in pairs(model:findModules(name)) do
			v.weight:fill(1)
			v.bias:zero()
		end
	end

	ConvInit('cudnn.SpatialConvolution')
	ConvInit('nn.SpatialConvolution')
	BNInit('fbnn.SpatialBatchNormalization')
	BNInit('cudnn.SpatialBatchNormalization')
	BNInit('nn.SpatialBatchNormalization')
	for k,v in pairs(model:findModules('nn.Linear')) do
		v.bias:zero()
	end
	model:cuda()

	if opt.cudnn == 'deterministic' then
		model:apply(function(m)
			if m.setMode then m:setMode(1,1,1) end
		end)
	end

	model:get(1).gradInput = nil

	return model
end

return createModel
