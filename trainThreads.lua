----------------------------------------------------------------------
-- SGD training of the network
----------------------------------------------------------------------

require 'optim'
require 'xlua'
require 'cutorch'
require 'cunn'
require 'VideoOptFlow'

local threads = require 'threads'

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
  cmd:option('-workers',2,'threads for asynchronous loading/training')
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

printFreq = 10 -- freq to print confusion matrix during epoch

paths.dofile('dataset.lua')
loader = dataLoader(opt.fps,opt.datapath)
do
  local gloader = dataLoader(opt.fps,opt.datapath)
  pool = threads.Threads(opt.workers,
    function ()
      require 'torch'
      paths.dofile('dataset.lua')
    end,
    function ()
      threadLoader = gloader
    end 
  )
end

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
  
  print("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
  print("==> online epoch # " .. epoch)
  for t = 1,#loader.trainIndeces do
    
    -- progress bar
    xlua.progress(t, #loader.trainIndeces)
    
    pool:addjob(
      function(idx)
        local inputCPU, labelsCPU = threadLoader:get(idx)
        return inputCPU, labelsCPU
      end,
      function(inputCPU,labelsCPU)
        if #inputCPU > 1 then
          local labels = labelsCPU:cuda()
          local input = {}
          for i = 1,#inputCPU do
            input[i] = inputCPU[i]:cuda()
          end
          
          local feval = function(x)
            -- append optical flow to input data 
            -- note: can't run optical flow calcs in thread callback since it uses the gpu
            --       the workers may write to overlapping memory on the gpu
            if opt.optflow then
              input = VideoOptFlow(input)
            end
            --print(input)
            
            gradParameters:zero()
            model:forget()
            
            local output = model:forward(input)
            local err = criterion:forward(output,labels)
            
            for i = 1,#output do
              confusion:add(output[i],labels[i])
            end
            
            local gradOutputs = criterion:backward(output,labels)
            model:backward(input,gradOutputs)
            gradParameters:div(#input)
            err = err/#input
            return err,gradParameters
          end
          optimMethod(feval,parameters,optimState)
          model:forget()
        end
      end,
      loader.trainIndeces[shuffle[t]]
    ) 
    --[[
    -- print updates periodically throughout epoch
    if t%printFreq==0 then
      --print(confusion)
      confusion:updateValids()
      print('mean class accuracy = ' .. confusion.totalValid*100 .. '%')
    end
    --]]
    
  end
  
  -- finish all threads training before continuing
  pool:synchronize()
    
  -- time taken
  time = sys.clock()-time
  print("\n==> training time = " .. (time*1000) .. 'ms')
  
  --print(confusion)
  confusion:updateValids()
  print('Average row correct: ' .. (confusion.averageValid*100) .. '%')
  print('Average rowUcol correct (VOC measure): ' .. (confusion.averageUnionValid*100) .. '%')
  print('Global correct: ' .. (confusion.totalValid*100) .. '%')
  
  -- update training log
  trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid*100}

  -- return global accuracy
  return 100*confusion.totalValid
end
  

























