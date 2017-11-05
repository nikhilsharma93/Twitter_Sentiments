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
local trainLogger = optim.Logger(paths.concat(opt.save, 'trainV1.log'))
-- Confusion Matrix
local confusion = optim.ConfusionMatrix(classes)

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> flattening model parameters')
-- Retrieve parameters and gradients:
-- this extracts and flattens all the trainable parameters of the mode
-- into a 1-dim vector
local w,dE_dw = model:getParameters()

----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> configuring optimizer')

local optimState = {
   learningRate = opt.learningRate,
   momentum = opt.momentum,
   weightDecay = opt.weightDecay,
   learningRateDecay = opt.learningRateDecay
}


----------------------------------------------------------------------
print(sys.COLORS.red ..  '==> defining training procedure')

local epoch

local function train(trainSet)

   confusion:zero()
   model:training()

   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- Small snippet to be able to change the learning rate at will
   if epoch % 1 == 0 then
     print (sys.COLORS.blue..'Change Learning Rate?')
     local handle = io.popen("bash readEpochChange.sh")
     local content = handle:read("*a")
     handle:close()
     if string.sub(content,1,3) == "yes" then
       print (sys.COLORS.blue .. 'Chaning Learning Rate')
       optimState['learningRate'] = optimState['learningRate']/10.0
     else
       print (sys.COLORS.blue .. 'Learning Rate Unchanged')
     end
   end


   -- do one epoch
   print(sys.COLORS.green .. '\n==> doing epoch on training data:')
   print ('Learning Rate for Epoch '..tostring(epoch)..': '..tostring(optimState['learningRate']))
   print("==> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

   for i, inputs, targets in trainSet:sampleiter(opt.batchSize, trainSet:size()) do

      -- disp progress
      xlua.progress(i, trainSet:size())
      collectgarbage()


      -- create closure to evaluate f(X) and df/dX
      local eval_E = function(w)
         -- reset gradients
         dE_dw:zero()

         -- evaluate function for complete mini batch
         local y = model:forward(inputs)

         local E = loss:forward(y,targets)

         -- estimate df/dW
         local dE_dy = loss:backward(y,targets)
         model:backward(inputs,dE_dy)

         -- Update confusion matrix
         confusion:batchAdd(y, targets)

         -- clip gradients
         dE_dw:clamp(-2.0, 2.0)

         -- return f and df/dX
         return E,dE_dw
      end

      -- optimize on current mini-batch
      optim.adam(eval_E, w, optimState)
   end

   confusion:updateValids()

   -- time taken
   time = sys.clock() - time
   time = time / trainSet:size()
   print("\n==> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)

   -- update logger/plot
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}



   if epoch % 1 == 0 then
     print (sys.COLORS.blue..'Save Model?')
     local handle = io.popen("bash readEpochChange.sh")
     local content = handle:read("*a")
     handle:close()
     if string.sub(content,1,3) == "yes" then
       print (sys.COLORS.blue .. 'Saving Model')
       local filename = 'results/modelV1.t7'
       --os.execute('mkdir -p ' .. sys.dirname(filename))
       print('==> saving model to '..filename)
       model1 = model:clone()
       --netLighter(model1)
       --torch.save(filename, model1)
       torch.save(filename, model1:clearState())
     else
       print (sys.COLORS.blue .. 'Did Not Save The Model')
     end
   end

   -- next epoch
   epoch = epoch + 1
end

-- Export:
return train
