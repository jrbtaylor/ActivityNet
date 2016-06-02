-- takes a video (table of frames) and performs local subtractive normalization
require 'nn'
require 'image'

function VideoNormalize(x)
  kernel = image.gaussian1D({size=15,normalize=true})
  for t = 1,#x do
    if torch.type(x[t])=='table' then
      x[t][1] = nn.SpatialSubtractiveNormalization(3,kernel):forward(x[t][1])
    else
      x[t] = nn.SpatialSubtractiveNormalization(3,kernel):forward(x[t])
    end
  end
  return x
end