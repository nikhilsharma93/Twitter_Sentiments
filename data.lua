------------------------------------------------------------------------------
-- __Author__ = Nikhil Sharma
------------------------------------------------------------------------------
require 'torch'
require 'nn'

local dl = require 'dataload'


-- Data parameters
local minFreq = 10 --Min freq
local seqLen = 50 --Length of sequence (context)
local dataDir = 'data/'--'/home/nikhil/Twitterdata/' --Path of data
local validRatio = 0.15 --Percentage of training data reserved for validation

classes = {'Negative', 'Positive'}

trainSet, validSet, testSet = dl.loadSentiment140(dataDir, minFreq, seqLen,
                                                  validRatio)



-- return package:
return {
  trainSet = trainSet,
  validSet = validSet,
  testSet = testSet,
  classes = classes,
  trainVocabLength = #trainSet.ivocab
}
