-- main file for activitynet challenge

require 'cutorch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
w_init = dofile '/home/rain/jtaylor/Documents/Torch_Workspace/torch-toolbox/weight-init.lua'

-- Terminal args
--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cmd = torch.CmdLine()
cmd:text()
cmd:text('ActivityNet Challenge')
-- sgd:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-LR', 1e-5, 'learning rate at t=0')
cmd:option('-LRDecay', 1e-7, 'learning rate decay')
cmd:option('-momentum', 0.9, 'momentum')
cmd:option('-weightDecay',1e-10,'weight decay')
cmd:option('-save','/usr/local/data/jtaylor/Deep/save','save directory')
cmd:option('-nEpochs',1000,'number of epochs')
-- data:
cmd:option('-workers',3,'threads for asynchronous loading/training')
cmd:option('-cropSize',128,'size for random crops for frames (h=w)')
cmd:option('-maxLength',3,'maximumum length of video clip to load (seconds)')
cmd:option('-fps',10,'frames per second')
cmd:option('-datapath','/usr/local/data2/jtaylor/Databases/ActivityNet/','path to data')
cmd:option('-nDonkeys',0,'parallel data-loaders')
-- model:
cmd:option('-rho',5,'backprop steps through time')
cmd:option('-optflow',false,'append optical flow to input data')

opt = cmd:parse(arg or {})

torch.manualSeed(opt.seed)
torch.setdefaulttensortype('torch.FloatTensor')

-- 
--%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
paths.dofile('model.lua')
if opt.workers>1 then
  paths.dofile('trainThreads.lua')
else
  paths.dofile('train.lua')
end
paths.dofile('validation.lua')
--paths.dofile('test.lua')

-- initialize weights
model = w_init(model,'kaiming')

epoch = 1
bestval = 0;
stall = 0;
stall_limit = 20;

for i=1,opt.nEpochs do
  train()
  valacc = validate()
  epoch = epoch+1
  if valacc > bestval then
    bestval = valacc
    stall = 0
    -- save/log current net
    print('==> saving model to ' .. opt.save)
    collectgarbage()
    model:clearState() -- saves a lot of space
    torch.save(paths.concat(opt.save,'model_' .. epoch .. '.t7'),model)
  else
    stall = stall+1
  end  
end






