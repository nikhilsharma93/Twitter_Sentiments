require 'rnn'
require 'nn'

local d = require 'data'

-- parameters
local vocabLength = d.trainVocabLength
local numberOfClasses = #d.classes
local inputDim = 100 -- embeddingSize
local hiddenDim = 4 -- number of hidden layers



-- Model
model = nn.Sequential()

-- lookup table
local lookup = nn.LookupTableMaskZero(vocabLength, inputDim)
  model:add(lookup)
  model:add(nn.SplitTable(1,2))

-- LSTM
stackLstm = nn.Sequential()
for i=1,hiddenDim do
  local lstm = nn.FastLSTM(inputDim, hiddenDim)
  stackLstm:add(lstm)
  inputDim = hiddenDim
end

-- add it to a Sequencer
model:add(nn.Sequencer(stackLstm))

-- select the last output alone
model:add(nn.SelectTable(-1))
model:add(nn.Linear(hiddenDim, numberOfClasses))
model:add(nn.LogSoftMax())


-- Loss function
local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())


-- return package:
return {
   model = model,
   loss = criterion,
}
