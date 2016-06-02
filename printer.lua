local printer, parent = torch.class('nn.printer', 'nn.Module')

function printer:__init(name)
  parent.__init(self)
  self._name = name or ''
  self._input = torch.Tensor()
  self._gradOutput = torch.Tensor()
end

function printer:updateOutput(input)
  print(self._name .. ':')
  if type(input)=='table' then
    print(input)
  else
    print(input)
  end
  self.output = input
  return self.output
end

function printer:updateGradInput(input, gradOutput)
   --self.gradInput:viewAs(gradOutput, input)
   self.gradInput = gradOutput
   return self.gradInput
end