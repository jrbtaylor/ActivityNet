cv = require 'cv'
require 'cv.cudaoptflow'
require 'cv.imgproc'
require 'cutorch'
require 'dpnn'

function VideoOptFlow(x)
  local optflow = cv.cuda.BroxOpticalFlow{}
  local flow = {}
  local im0, im1, f
  
  local function preprocess(x)
    x = nn.Convert('chw','hwc'):forward(x)
    local y = cv.cvtColor{x,y,cv.COLOR_RGB2GRAY}:cuda()
    y = y/y:max()
    return y
  end
  
  for t = 1,#x-1 do
    im0 = preprocess(x[t])
    im1 = preprocess(x[t+1])
    f = optflow:calc{I0=im0,I1=im1}
    f = nn.Convert('hwc','chw'):forward(f):cuda()
    table.insert(flow,f)
  end
  if #flow>0 then
    -- copy last frame of optical flow to make it the same length as video (assume motion is coherent)
    table.insert(flow,flow[#flow])
  else
    print('warning, single-frame video loaded')
    flow = {torch.CudaTensor():zeros(x[#x]:size())}
  end
  -- append optical flow to input in a table
  local y = {}
  for t = 1,#x do
    table.insert(y,{x[t],flow[t]})
  end
  return y
end