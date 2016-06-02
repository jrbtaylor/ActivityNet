----------------------------------------------------------------------
-- Validation of network
-- needs to be run after training script (multi-threaded)
----------------------------------------------------------------------

-- parse command line arguments
if not opt then
  print '==> processing options'
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')
  cmd:option('-save', '/usr/local/data/jtaylor/Deep/save', 'subdirectory to save/log experiments in')
  cmd:option('-workers',2,'threads for asynchronous loading/training')
  cmd:text()
  opt = cmd:parse(arg or {})
end

-- logs
valLogger = optim.Logger(paths.concat(opt.save, 'val.log'))

function validate()
    
  -- time stuff
  local time = sys.clock()
  
  epoch = epoch or 1
  confusion = optim.ConfusionMatrix(loader.classes)
  
  -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
  model:evaluate()
  --normstab = model:findModules('nn.NormStabilizer')
  --for ns = 1,#normstab do
  --  normstab[ns]:training()
  --end
    
  for t = 1,#loader.valIndeces do
    
    -- progress bar
    xlua.progress(t, #loader.valIndeces)
    
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
          
          -- append optical flow to input data 
          -- note: can't run optical flow calcs in thread callback since it uses the gpu
          --       the workers may write to overlapping memory on the gpu
          input = VideoOptFlow(input)
          
          model:forget()
          
          local output = model:forward(input)
          
          for i = 1,#output do
            confusion:add(output[i],labels[i])
          end

        end
      end,
      loader.valIndeces[t]
    )    
  end
    
  -- time taken
  time = sys.clock()-time
  print("\n==> validation time = " .. (time*1000) .. 'ms')
  
  --print(confusion)
  confusion:updateValids()
  print('Average row correct: ' .. (confusion.averageValid*100) .. '%')
  print('Average rowUcol correct (VOC measure): ' .. (confusion.averageUnionValid*100) .. '%')
  print('Global correct: ' .. (confusion.totalValid*100) .. '%')
  
  -- update log
  valLogger:add{['% mean class accuracy (validation set)'] = confusion.totalValid*100}

   -- return global accuracy
  return 100*confusion.totalValid
end
  

























