require 'utils'

function initializeParameters(hiddenSize, visibleSize)
    ---- Initialize parameters randomly based on layer sizes.
    local r  = math.sqrt(6) / math.sqrt(hiddenSize+visibleSize+1)   -- we'll choose weights uniformly from the interval [-r, r]
    local W1 = torch.rand(hiddenSize, visibleSize) * 2 * r - r
    local W2 = torch.rand(visibleSize, hiddenSize) * 2 * r - r

    local b1 = torch.Tensor(hiddenSize, 1):zero()
    local b2 = torch.Tensor(visibleSize, 1):zero()

    -- Convert weights and bias gradients to the vector form.
    -- This step will "unroll" (flatten and concatenate together) all 
    -- your parameters into a vector, which can then be used with minFunc. 
    local hv = hiddenSize*visibleSize

    local theta = ncat(W1:reshape(hv), W2:reshape(hv), b1:t()[1], b2:t()[1]);

    return theta
end

