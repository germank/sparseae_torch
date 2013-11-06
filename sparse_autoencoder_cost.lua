require 'utils'

function sparseAutoencoderCost(theta, visibleSize, hiddenSize,
                                lambda, sparsityParam, beta, data)
    -- visibleSize: the number of input units (probably 64) 
    -- hiddenSize: the number of hidden units (probably 25) 
    -- lambda: weight decay parameter
    -- sparsityParam: The desired average activation for the hidden units (denoted in the lecture
    --                           notes by the greek alphabet rho, which looks like a lower-case "p").
    -- beta: weight of sparsity penalty term
    -- data: Our 64x10000 matrix containing the training data.  So, data[i] is the i-th training example. 
      
    -- The input theta is a vector (because minFunc expects the parameters to be a vector). 
    -- We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
    -- follows the notation convention of the lecture notes. 
    local W1 = theta[{{1,hiddenSize*visibleSize}}]:reshape(hiddenSize, visibleSize)
    local W2 = theta[{{hiddenSize*visibleSize+1,2*hiddenSize*visibleSize}}]:reshape(visibleSize, hiddenSize)
    local b1 = theta[{{2*hiddenSize*visibleSize+1,2*hiddenSize*visibleSize+hiddenSize}}]:reshape(
                       hiddenSize,1)
    local b2 = theta[{{2*hiddenSize*visibleSize+hiddenSize+1,(#theta)[1]}}]:reshape(visibleSize,1)

    -- Cost and gradient variables (your code needs to compute these values). 
    -- Here, we initialize them to zeros. 
    local cost = 0
    local W1grad = torch.Tensor(W1:size()):zero() 
    local W2grad = torch.Tensor(W2:size()):zero()
    local b1grad = torch.Tensor(b1:size()):zero() 
    local b2grad = torch.Tensor(b2:size()):zero()

    ---- ---------- YOUR CODE HERE --------------------------------------
    --  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
    --                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
    --
    -- W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
    -- Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
    -- as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
    -- respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
    -- with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
    -- [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
    -- of the lecture notes (and similarly for W2grad, b1grad, b2grad).
    -- 
    -- Stated differently, if we were using batch gradient descent to optimize the parameters,
    -- the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 

   
    local z2 = W1 * data + b1:expand(b1:size()[1],data:size()[2])
    local a2 = sigmoid(z2)
    local z3 = W2 * a2 + b2:expand(b2:size()[1], data:size()[2])
   
    local a3 = sigmoid(z3) --


    cost = torch.mean(torch.norm(a3 - data, 2, 2)) --norm2 over dimension 2
      
    local d3 = torch.cmul(-(data - a3) ,torch.cmul(a3, -a3 + 1))
   
    W2grad = d3 * a3:t()/data:size()[2] --?
    b2grad = torch.mean(d3, 2)
   
    local d2 = torch.cmul(W2:t() * d3, torch.cmul(a2, -a2 + 1))

    W1grad = d2 * a2:t()/data:size()[2] --?
    b1grad = torch.mean(d2, 2)
   


    ---------------------------------------------------------------------
    -- After computing the cost and gradient, we will convert the gradients back
    -- to a vector format (suitable for minFunc).  Specifically, we will unroll
    -- your gradient matrices into a vector.

    local grad = ncat(flatten(W1grad), flatten(W2grad), flatten(b1grad), flatten(b2grad))
    --collect the garbage before we leave
    collectgarbage()
    return cost, grad
end
---------------------------------------------------------------------
-- Here's an implementation of the sigmoid function, which you may find useful
-- in your computation of the costs and the gradients.  This inputs a (row or
-- column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigmoid(x)
   --GermanK: I think there should be a more efficient way to write it
   --but I'm not sure how to write it.
   --return x:apply(function(x_i) return 1 / (1 + math.exp(-x_i)) end);
   return torch.pow(torch.exp(-x) + 1, -1)
end

