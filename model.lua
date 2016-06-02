require 'nn'
require 'rnn'
require 'dpnn'
require 'cunn'
require 'cutorch'
require 'optim'
require 'nngraph'
--nn.FastLSTM.usenngraph = true -- faster?
dofile '/home/rain/jtaylor/Documents/Torch_Workspace/visatt/debugger.lua'

opt = opt or {}
opt.rho = opt.rho or 5 -- backprop through time steps

--------------------------------------------------------------------------

local function relu()
  return nn.ReLU(true)
end
local function shortcutAdd(mod)
  return nn.Sequential()
    :add(nn.ConcatTable()
      :add(mod)
      :add(nn.Identity())
    )
    :add(nn.CAddTable())
end
local function shortcutConcat(mod)
  return nn.Sequential()
    :add(nn.ConcatTable()
      :add(mod)
      :add(nn.Identity())
    )
    :add(nn.JoinTable(2))
end
local function convBlock(fin,fout)
  local conv = nn.Sequential()
    :add(nn.SpatialConvolution(fin,fout,3,3,1,1,1,1))
    :add(nn.SpatialCrossMapLRN(fout))
    :add(relu())
    :add(nn.SpatialConvolution(fout,fout,3,3,1,1,1,1))
    :add(nn.SpatialCrossMapLRN(fout))
  if fin ~= fout then
    local shortcut = nn.Sequential()
      :add(nn.SpatialConvolution(fin,fout,1,1))
      :add(nn.SpatialCrossMapLRN(fout))
  else
    local shortcut = nn.Identity()
  end
  return nn.Sequential()
    :add(nn.ConcatTable()
      :add(conv)
      :add(shortcut)
    )
    :add(nn.CAddTable())
    :add(relu())
end 
local function max()
  return nn.SpatialMaxPooling(2,2,2,2)
end

require 'loadcaffe'
require 'matio'
modeldir = '/usr/local/data/jtaylor/Pretrained_Nets/VGG_ILSVRC_19/'
prototxt = modeldir .. 'VGG_ILSVRC_19_layers_deploy.prototxt'
binary = modeldir .. 'VGG_ILSVRC_19_layers.caffemodel'
imnetCNN = loadcaffe.load(prototxt,binary)
-- note: model takes 224x224 BGR crops
remove = 10 -- final layers to remove
for l = 1,remove do
  imnetCNN:remove()
end
-- swap r and b channels of weights in the first layer (because who uses BGR???)
r = imnetCNN:get(1).weight[{{},1,{},{}}]:clone() -- clone is needed to avoid just making pointer
imnetCNN:get(1).weight[{{},1,{},{}}] = imnetCNN:get(1).weight[{{},3,{},{}}]
imnetCNN:get(1).weight[{{},3,{},{}}] = r
r = nil -- delete the copy
  
-- overload accGradParameters to fix the weights (~20% faster than not fixing them)
imnetCNN.accGradParameters = function(self) end

-- LSTM for high-level motion features
fmot = 256
--motionNet = nn.Recurrent(fmot,nn.Linear(512,fmot),nn.FastLSTM(fmot,fmot,opt.rho),nn.ReLU(),opt.rho)
RNN1 = nn.Sequential()
  :add(nn.Recurrent(fmot,nn.Linear(512,fmot),nn.Linear(fmot,fmot),nn.ReLU(),opt.rho))
  --:add(nn.debugger('rnn1'))
  :add(nn.NormStabilizer())
  :add(nn.Recurrent(fmot,nn.Linear(fmot,fmot),nn.Linear(fmot,fmot),nn.ReLU(),opt.rho))
--RNN2 = nn.Sequential()
--  :add(nn.Recurrent(4096,nn.Linear(4096,4096),nn.Linear(4096,4096),nn.ReLU(),opt.rho))

appearanceNet = nn.Sequential()
  :add(imnetCNN)
  :add(nn.Collapse(2))
  :add(nn.Convert('hw','wh'))
  :add(shortcutConcat(RNN1))
  :add(nn.Convert('wh','hw'))
  :add(nn.View(-1))
  :add(nn.Linear((512+fmot)*8*8,4096))
 -- :add(shortcutAdd(RNN2))
  :add(nn.ReLU())
  
if opt.optflow then
  RNNo = nn.Sequential()
    :add(nn.Recurrent(fmot,nn.Linear(128,fmot),nn.Linear(fmot,fmot),nn.ReLU(),opt.rho))
    --:add(nn.debugger('rnn2'))
    :add(nn.NormStabilizer())
    :add(nn.Recurrent(fmot,nn.Linear(fmot,fmot),nn.Linear(fmot,fmot),nn.ReLU(),opt.rho))

  optflowNet = nn.Sequential()
    --:add(nn.SelectTable(2))
    :add(nn.SpatialConvolution(2,16,7,7)) -- 122x122
    :add(nn.ReLU())
    :add(nn.SpatialMaxPooling(3,3,2,2)) -- 60x60
    :add(nn.SpatialDropout(0.2))
    :add(nn.SpatialConvolution(16,32,5,5)) -- 56x56
    :add(nn.ReLU())
    :add(nn.SpatialMaxPooling(3,3,2,2)) -- 27x27
    :add(nn.SpatialDropout(0.3))
    :add(nn.SpatialConvolution(32,64,3,3)) -- 25x25
    :add(nn.ReLU())
    :add(nn.SpatialMaxPooling(3,3,2,2)) -- 12x12
    :add(nn.SpatialDropout(0.4))
    :add(nn.SpatialConvolution(64,128,3,3)) -- 10x10
    :add(nn.ReLU())
    :add(nn.SpatialMaxPooling(3,3,2,2)) -- 4x4
    :add(nn.Collapse(2))
    :add(nn.Convert('hw','wh'))
    :add(shortcutConcat(RNNo))
    :add(nn.Convert('wh','hw'))
    :add(nn.View(-1))
    :add(nn.Dropout(0.5))
    :add(nn.Linear((128+fmot)*4*4,1024))
    :add(nn.ReLU())
    
  parallelNet = nn.Sequential()
    :add(nn.ParallelTable()  
      :add(appearanceNet)
      :add(optflowNet)
    )
    :add(nn.JoinTable(1))
  
  outsize = 4096+1024
else
  parallelNet = appearanceNet
  outsize = 4096
end

finalNet = nn.Sequential()
  :add(parallelNet)
  :add(nn.Dropout(0.5))
  :add(nn.Linear(outsize,272)) -- 272
  :add(nn.LogSoftMax())

model = nn.Sequencer(finalNet)
criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

print('=> Model')
print(model)
print('=> Criterion')
print(criterion)

criterion:cuda()
model:cuda()

collectgarbage()

