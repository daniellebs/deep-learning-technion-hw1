local mnist = require 'mnist';

local trainData = mnist.traindataset().data:float();
local trainLabels = mnist.traindataset().label:add(1);
testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);

--We'll start by normalizing our data
local mean = trainData:mean()
local std = trainData:std()
trainData:add(-mean):div(std); 
testData:add(-mean):div(std);

require 'nn'
require 'cunn'


---	 ### predefined constants

require 'optim'
batchSize = 32

optimState = {
    learningRate = 0.1
    
}

criterion = nn.ClassNLLCriterion():cuda()

--- ### Main evaluation + training function


function forwardNet(data, labels)

    local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
    
    for i = 1, data:size(1) - batchSize, batchSize do
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        confusion:batchAdd(y,yt)     
    end
    
    confusion:updateValids()
    local avgError = 1 - confusion.totalValid

    return avgError
end


function testModel()
    -- Load trained net (the model)
    model = torch.load("trained_model") 

    testError = forwardNet(testData, testLabels)
    return testError
end


testError = testModel()
print('Test error: ' .. testError)