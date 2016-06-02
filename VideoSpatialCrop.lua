-- like SpatialUniformCrop, grabs a random oheight x owidth patch
-- but the same patch for all frames of a video

require 'nn'

local VideoSpatialCrop, parent = torch.class("nn.VideoSpatialCrop", "nn.Module")

function VideoSpatialCrop:__init(oheight, owidth)
   parent.__init(self)
   self.oheight = oheight
   self.owidth = owidth or oheight
end

function VideoSpatialCrop:updateOutput(input)
  if type(input)=='table' then -- each frame is a separate entry in a table, batchsize is 1
    if #input>0 then
      local iH, iW = input[1]:size(2), input[1]:size(3)
      if self.train ~= false then -- random crop for training
        h1 = math.ceil(torch.uniform(1e-2,iH-self.oheight))
        w1 = math.ceil(torch.uniform(1e-2,iW-self.owidth))
      else -- center crop for testing
        h1 = math.ceil((iH-self.oheight)/2)
        w1 = math.ceil((iW-self.owidth)/2)
      end
      self.output = {}
      self.coord = {h1,w1}
      for i=1,#input do
        self.output[i] = input[i]:narrow(2,h1,self.oheight):narrow(3,w1,self.owidth)
      end
    else
      self.output = {}
    end
  else
    -- if input is batchsize 1, dim 4 (channel x time x height x width)
    -- then toBatch(input,4) returns a dim 5 tensor (batch=1 x channel x time x height x width)
    input = self:toBatch(input, 4)

    self.output:resize(input:size(1), input:size(2), input:size(3), self.oheight, self.owidth)
    self.coord = self.coord or torch.IntTensor()
    self.coord:resize(input:size(1), 2)

    local iH, iW = input:size(4), input:size(5)
    if self.train ~= false then
     for i=1,input:size(1) do
        -- do random crop
        local h1 = math.ceil(torch.uniform(1e-2, iH-self.oheight))
        local w1 = math.ceil(torch.uniform(1e-2, iW-self.owidth))
        local crop = input[i]:narrow(3,h1,self.oheight):narrow(4,w1,self.owidth)
        self.output[i]:copy(crop)
        -- save crop coordinates for backward
        self.coord[{i,1}] = h1
        self.coord[{i,2}] = w1
     end
    else
      -- use center crop
      local h1 = math.ceil((iH-self.oheight)/2)
      local w1 = math.ceil((iW-self.owidth)/2)
      local crop = input:narrow(4,h1,self.oheight):narrow(5,w1,self.owidth)
      self.output:copy(crop)
    end

    self.output = self:fromBatch(self.output, 1)
  end
  return self.output
end

function VideoSpatialCrop:updateGradInput(input, gradOutput)
   input = self:toBatch(input, 4)
   gradOutput = self:toBatch(gradOutput, 4)
   
   self.gradInput:resizeAs(input):zero()
   if self.scale ~= nil then
      local iH, iW = input:size(4), input:size(5)
      for i=1,input:size(1) do
         local s = self.scales[i]
         local soheight = math.ceil(s*self.oheight)
         local sowidth = math.ceil(s*self.owidth)

         local h, w = self.coord[{i,1}], self.coord[{i,2}]
        
         local ch = math.ceil(iH/2 - (iH-soheight)/2 + h)
         local cw = math.ceil(iW/2 - (iH-sowidth)/2 + w)

         local h1 = ch - math.ceil(soheight/2)
         local w1 = cw - math.ceil(sowidth/2)
         if h1 < 1 then h1 = 1 end
         if w1 < 1 then w1 = 1 end

         local crop = input[i]:narrow(3, h1, soheight):narrow(4, w1, sowidth)
         local samplerGradInput = self.scaler:updateGradInput(crop, gradOutput[i])

         self.gradInput[i]:narrow(3, h1, soheight):narrow(4, w1, sowidth):copy(samplerGradInput)
      end
   else
      for i=1,input:size(1) do
         local h1, w1 = self.coord[{i,1}], self.coord[{i,2}]
         self.gradInput[i]:narrow(3,h1,self.oheight):narrow(4,w1,self.owidth):copy(gradOutput[i])
      end
   end
   
   self.gradInput = self:fromBatch(self.gradInput, 1)
   return self.gradInput
end

function VideoSpatialCrop:type(type, cache)
   self.coord = nil
   return parent.type(self, type, cache)
end
