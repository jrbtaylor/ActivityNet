-- takes a video (table of frames) and horizontally flips each frame

require 'image'
require 'dpnn'

function VideoHFlip(x)
  if torch.uniform()>0.5 and #x>0 then
    for t = 1,#x do
      x[t] = image.hflip(x[t])
    end
  end
  return x
end
    
 --[[ 
  -- make it 5D by padding singleton dim
  x = nn.Module:toBatch(x,4) -- gross abuse of OOP by I need this function...
    
  -- flippity flip flip
  for b = 1,x:size(1) do
    -- flip with only 0.5 probability
    if torch.uniform()>0.5 then
      -- swap time and channel since image.hflip will do <= 3 channels only
      frames = nn.Convert('cbhw','bchw'):forward(x[b]) -- now time x channel x h x w
      for f = 1,frames:size(1) do
        frames[f] = image.hflip(frames[f])
      end
      -- swap back time and channel
      frames = nn.Convert('bchw','cbhw'):forward(frames)
      -- and assign back to x
      x[b] = frames
    end
  end
return x
end
--]]