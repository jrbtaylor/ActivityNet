torch.setdefaulttensortype('torch.FloatTensor')
require 'sys'
require 'xlua'
require 'image'
require 'ffmpeg'
local torchx = require 'torchx'
local js = require 'json'
local paths = require 'paths'
require 'VideoSpatialCrop'
require 'VideoHFlip'
require 'VideoNormalize'

opt = opt or {}
opt.cropSize = opt.cropSize or 128
opt.maxLength = opt.maxLength or 5 -- maximum length to load in seconds

---------------------------------------------------------------------------
local dataset = torch.class('dataLoader')

--==========================================================================================
-- constructor
--==========================================================================================
function dataset:__init(fps,datapath,jsonpath)
  
  self.fps = fps or 5
  local datapath = datapath or '/usr/local/data2/jtaylor/Databases/ActivityNet/'
  local jsonpath = jsonpath or (datapath .. 'activity_net.v1-3.min.json')
  
  -- load json file
  self.json = js.load(jsonpath)
  -- note: json['database'][filename (w/o ".mp4")] is syntax for indexing into lua object returned
  -- e.g. json['database']['31KEa5VhvPs'][resolution/duration/annotations/subset]
  -- for json: json['database'][filename][annotations][x][label/segment]
  --                                                       |   |
  --                                                       |   -->  segment[1]=start, segment[2]=end
  --                                                       |
  --                                                       --> can have multiple labels
  
  -- Need to also look at files in dir because many failed to download
  local fileTable = paths.indexdir(datapath .. 'videos/','mp4')
  self.filePaths = {}
  self.trainIndeces = {}
  self.valIndeces = {}
  self.testIndeces = {}
  self.annotations = {}
  self.duration = {}
  self.resolution = {}
  self.numSamples = fileTable:size()
  for f = 1,self.numSamples do
    -- save file path to load later by index
    table.insert(self.filePaths,fileTable:filename(f))
    local name = string.gsub(paths.basename(fileTable:filename(f),'.mp4'),'v_','')
    if self.json['database'][name] then
      if self.json['database'][name]['subset'] == 'training' then
        table.insert(self.trainIndeces,f)
      elseif self.json['database'][name]['subset'] == 'validation' then
        table.insert(self.valIndeces,f)
      elseif self.json['database'][name]['subset'] == 'testing' then
        table.insert(self.testIndeces,f)
      end
      table.insert(self.annotations,self.json['database'][name]['annotations'])
      table.insert(self.duration,self.json['database'][name]['duration'])
      table.insert(self.resolution,self.json['database'][name]['resolution'])
    end
  end
  
  -- Find class names and indeces
  -- note: completely destroys info on parent nodes, not sure how to use that in training anyway
  -- background (no action) is 1, others are 2-201
  self.classes = {'background'}
  self.classIndeces = {1}
  for k,v in pairs(self.json['taxonomy']) do
    self.classIndeces[v['nodeName']] = k+1
    table.insert(self.classes,v['nodeName'])
  end
  collectgarbage()
end


--==========================================================================================
-- Load a video at a given index in the dataset
--==========================================================================================
function dataset:get(ind)
  
  -- Restrict loading to random subsets up to a maximum length (or all, if shorter)
  self.duration[ind] = self.duration[ind] or 0
  local dur = self.duration[ind]
  dur = math.min(dur,opt.maxLength)
  
  -- Balance classes somewhat (action ~half the time, background ~half the time)
  local annot = self.annotations[ind] or {}
  if #annot>0 then -- sample around an action
    local a = torch.random(1,#annot) -- choose one if there's multiple annotations
    local t1 = annot[a].segment[1] -- action start time
    local t2 = annot[a].segment[2] -- action end time
    if torch.uniform(0,1)>0.5 then -- sample around action start
      start = torch.uniform(math.max(t1-dur,0),math.min(t1+dur,self.duration[ind]-dur))
    else -- sample around action end
      start = torch.uniform(math.max(t2-dur,0),math.min(t2+dur,self.duration[ind]-dur))
    end
  else
    start = torch.uniform(0,self.duration[ind]-dur)
  end
  
  -- Load the video into a tensor ----------------------------
  local h_out, w_out = opt.cropSize, opt.cropSize
  local res = self.resolution[ind] or '0x0'
  -- get the resolution from the string
  local x = string.find(res,'x')
  local h_in = tonumber(string.sub(res,x+1))
  local w_in = tonumber(string.sub(res,1,x-1))
  -- downsample by greater ratio of h_out/h_in or w_out/w_in
  if h_out/h_in > w_out/w_in then
    w_out = math.floor(w_in*h_out/h_in)
  else
    h_out = math.floor(h_in*w_out/w_in)
  end
  -- load video
  local video = ffmpeg.Video{path=self.filePaths[ind], width=w_out, height=h_out, fps=self.fps, length=dur, seek=start, silent=true}
  -- remove extra entries from table
  video[1]['path'] = nil
  video[1]['channel'] = nil
  video[1]['sformat'] = nil
  video = video[1] -- remove everything else
  
  if #video>1 then
    -- do random crop
    video = nn.VideoSpatialCrop(opt.cropSize,opt.cropSize):forward(video)
    
    -- horizontal flip (0.5 probability built in)
    video = VideoHFlip(video)
               
    -- local subtractive normalization
    video = VideoNormalize(video)
  end
  
  
  -- Format multiple labels to a vector ---------------------
  local labels = torch.FloatTensor():zeros(#video)+1
  if #video > 0 then
    for a = 1,#annot do
      local l = self.classIndeces[annot[a].label] -- index of class name for this annotation
      local t1 = annot[a].segment[1]-start -- start time of action (seconds)
      local t2 = annot[a].segment[2]-start -- end time of action (seconds)
      local i1 = math.ceil(t1*self.fps)
      local i2 = math.ceil(t2*self.fps)
      if i1<#video and i2>0 then -- if any portion of action overlaps with clip
        i1 = math.max(i1,1)
        i2 = math.min(i2,#video)
        labels:narrow(1,i1,i2-i1+1):fill(l) -- :fill(l) JUST DETECTION RIGHT NOW
      end
    end
  end
  
  collectgarbage()
  return video, labels
end


--==========================================================================================
-- Sample a single video from the dataset
--==========================================================================================
function dataset:sample()
  local ind = torch.random(1,#self.trainIndeces)
  local data, label = self:get(ind)
  collectgarbage()
  return data, label
end


return dataset


