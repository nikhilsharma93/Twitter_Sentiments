------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------

require 'torch'
require 'xlua'
require 'optim'
require 'os'
require 'rnn'
require 'nn'

----------------------------------------------------------------------
-- Model + Loss + Data:
local t = require 'model'
local model = t.model
local loss = t.loss


-- Log results to files
local testLogger = optim.Logger(paths.concat(opt.save, 'testV1.log'))
-- Confusion Matrix
local confusion = optim.ConfusionMatrix(classes)


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining testing procedure')

local epoch

local function test(testSet)
   confusion:zero()
   model:evaluate()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()


   -- do one epoch
   print(sys.COLORS.green .. '\n==> doing epoch on testing data:')
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

   for i, inputs, targets in testSet:sampleiter(batchSize, testSet:size()) do

      -- disp progress
      xlua.progress(i, testSet:size())
      collectgarbage()

     -- evaluate function for complete mini batch
     local y = model:forward(inputs)

     local E = loss:forward(y,targets)

     -- Update confusion matrix
     confusion:batchAdd(y, targets)

   end

   confusion:updateValids()

   -- time taken
   time = sys.clock() - time
   time = time / testSet:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}

   -- next epoch
   epoch = epoch + 1
end

-- Export:
return test
