----------------------------------------------------------------------
-- SGD training of the network
----------------------------------------------------------------------

require 'optim'
require 'xlua'
require 'cutorch'
require 'cunn'

----------------------------------------------------------------------

-- parse command line arguments
if not opt then
  print '==> processing options'
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')
  cmd:option('-save', '/usr/local/data/jtaylor/Deep/save', 'subdirectory to save/log experiments in')
  cmd:option('-LR', 1e-3, 'learning rate at t=0')
  cmd:option('-LRDecay', 1e-5, 'learning rate decay')
  cmd:option('-momentum', 0.9, 'momentum')
  cmd:option('-weightDecay',1e-7,'weight decay')
  cmd:text()
  opt = cmd:parse(arg or {})
end

-- training logs
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))

-- get model parameters
parameters,gradParameters = model:getParameters()

-- configure SGD
optimState = {
  learningRate = opt.LR,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum,
  learningRateDecay = opt.LRDecay
}
optimMethod = optim.sgd

paths.dofile('dataset.lua')
loader = dataLoader(opt.fps,opt.datapath)

printFreq = 10 -- freq to print confusion matrix during epoch

function train()
    
  -- time stuff
  local time = sys.clock()
  
  
  epoch = epoch or 1
  --confusion = optim.ConfusionMatrix({1,2})
  confusion = optim.ConfusionMatrix(loader.classes)
  
  -- set model to training mode (for modules that differ in training and testing, like Dropout)
  model:training()
  
  -- shuffle at each epoch
  shuffle = torch.randperm(#loader.trainIndeces)
  
  print("==> online epoch # " .. epoch)
  for t = 1,#loader.trainIndeces do
    
    -- progress bar
    xlua.progress(t, #loader.trainIndeces)

    -- load data and labels
    local inputCPU, labelsCPU = loader:get(loader.trainIndeces[shuffle[t]])
    if #inputCPU > 0 then
      local labels = labelsCPU:cuda()
      local input = {}
      for i = 1,#inputCPU do
        input[i] = inputCPU[i]:cuda()
      end  

      -- train current sample
      local feval = function(x)        
        -- reset gradients
        gradParameters:zero()
        model:forget()

        -- forward pass
        local output = model:forward(input)
        local err = criterion:forward(output,labels)
        
        -- update confusion matrix
        -- requires tensor instead of table
        for i = 1,#output do
          confusion:add(output[i],labels[i])
        end

        -- backprop
        local gradOutputs = criterion:backward(output,labels)
        model:backward(input,gradOutputs)
        
        -- normalize
        gradParameters:div(#input)
        err = err/#input
        
        return err,gradParameters
      end
      optimMethod(feval,parameters,optimState)
      model:forget()
      collectgarbage()
    end
    
    -- print updates periodically throughout epoch
    if t%printFreq==0 then
      --print(confusion)
      confusion:updateValids()
      print('mean class accuracy = ' .. confusion.totalValid*100 .. '%')
    end
    
  end
    
  -- time taken
  time = sys.clock()-time
  print("\n==> training time = " .. (time*1000) .. 'ms')
  
  print(confusion)
  
  -- print accuracy update
  confusion:updateValids() -- necessary to get .totalValid without printing full matrix
  print('==> mean class accuracy = ' .. confusion.totalValid*100 .. '%')
  
  -- update training log
  trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid*100}

  -- save/log current net
  print('==> saving model to ' .. opt.save)
  collectgarbage()
  --model:clearState() -- saves a lot of space
  --torch.save(paths.concat(opt.save,'model_' .. epoch .. '.t7'),model)

  -- return global accuracy
  return 100*confusion.totalValid
end
  

























