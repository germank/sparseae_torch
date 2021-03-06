---- CS294A/CS294W Programming Assignment Starter Code

--  Instructions
--  ------------
-- 
--  This file contains code that helps you get started on the
--  programming assignment. You will need to complete the code in sample_images.lua,
--  sparse_autoencoder_cost.lua and compute_numerical_gradient.lua. 
--  For the purpose of completing the assignment, you do not need to
--  change the code in this file. 
--
----======================================================================
---- STEP 0: Here we provide the relevant parameters values that will
--  allow your sparse autoencoder to get good filters; you do not need to 
--  change the parameters below.
require 'torch'
visibleSize = 8*8;   -- number of input units 
hiddenSize = 25;     -- number of hidden units 
sparsityParam = 0.01;   -- desired average activation of the hidden units.
                     -- (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
             --  in the lecture notes). 
lambda = 0.0001;     -- weight decay parameter       
beta = 3;            -- weight of sparsity penalty term       

----======================================================================
---- STEP 1: Implement sampleIMAGES
--
--  After implementing sampleIMAGES, the display_network command should
--  display a random sample of 200 patches from the dataset

require 'sample_images'
IMAGES = torch.load('IMAGES.th7')
patches = sample_images(IMAGES, visibleSize);
require 'display_network'
function pick_random_n(M, n)
   M_sample={}
   for i=1,n do
      M_sample[#M_sample+1] = math.random(1,M:size()[1])
   end 
   return M_sample  
end
patch_sample=pick_random_n(patches, 200)
if qt then
    display_network(patches:index(1, torch.LongTensor(patch_sample)),8)
end


--  Obtain random parameters theta
require 'initialize_parameters'
theta = initializeParameters(hiddenSize, visibleSize);

----======================================================================
---- STEP 2: Implement sparseAutoencoderCost
--
--  You can implement all of the components (squared error cost, weight decay term,
--  sparsity penalty) in the cost function at once, but it may be easier to do 
--  it step-by-step and run gradient checking (see STEP 3) after each step.  We 
--  suggest implementing the sparseAutoencoderCost function using the following steps:
--
--  (a) Implement forward propagation in your neural network, and implement the 
--      squared error term of the cost function.  Implement backpropagation to 
--      compute the derivatives.   Then (using lambda=beta=0), run Gradient Checking 
--      to verify that the calculations corresponding to the squared error cost 
--      term are correct.
--
--  (b) Add in the weight decay term (in both the cost function and the derivative
--      calculations), then re-run Gradient Checking to verify correctness. 
--
--  (c) Add in the sparsity penalty term, then re-run Gradient Checking to 
--      verify correctness.
--
--  Feel free to change the training settings when debugging your
--  code.  (For example, reducing the training set size or 
--  number of hidden units may make your code run faster; and setting beta 
--  and/or lambda to zero may be helpful for debugging.)  However, in your 
--  final submission of the visualized weights, please use parameters we 
--  gave in Step 0 above.

require 'sparse_autoencoder_cost'
cost, grad = sparseAutoencoderCost(theta, visibleSize, hiddenSize, lambda, 
                                     sparsityParam, beta, patches:t());

----======================================================================
---- STEP 3: Gradient Checking
--
-- Hint: If you are debugging your code, performing gradient checking on smaller models 
-- and smaller training sets (e.g., using only 10 training examples and 1-2 hidden 
-- units) may speed things up.

-- First, lets make sure your numerical gradient computation is correct for a
-- simple function.  After you have implemented computeNumericalGradient.m,
-- run the following: 
require 'check_numerical_gradient'
checkNumericalGradient();

-- Now we can use it to check your cost function and derivative calculations
-- for the sparse autoencoder.  
require 'compute_numerical_gradient'
numgrad = computeNumericalGradient( function(x) return 
                                                sparseAutoencoderCost(x, visibleSize, 
                                                  hiddenSize, lambda, 
                                                  sparsityParam, beta, 
                                                  patches:t()) end, theta);

-- Use this to visually compare the gradients side by side
print(numgrad)
print(grad); 
-- Compare numerically computed gradients with the ones obtained from backpropagation
diff = torch.norm(numgrad-grad)/torch.norm(numgrad+grad);
print(diff); -- Should be small. In our implementation, these values are
            -- usually less than 1e-9.
            --
            -- When you got this working, Congratulations!!! 

----======================================================================
---- STEP 4: After verifying that your implementation of
--  sparseAutoencoderCost is correct, You can start training your sparse
--  autoencoder with minFunc (L-BFGS).

--  Randomly initialize the parameters
theta = initializeParameters(hiddenSize, visibleSize);

--  Use minFunc to minimize the function
require 'optim'
minFunc = optim.lbfgs     -- Here, we use L-BFGS to optimize our cost
                          -- function. Generally, for minFunc to work, you
                          -- need to send a function with two outputs: the
                          -- function value and the gradient. In our problem,
                          -- sparseAutoencoderCos satisfies this.
options = {}
options.maxIter = 400;      -- Maximum number of iterations of L-BFGS to run 


opttheta, cost = minFunc( function(x) return sparseAutoencoderCost(x, 
                                   visibleSize, hiddenSize, 
                                   lambda, sparsityParam, 
                                   beta, patches:t()) end, 
                              theta, options);

----======================================================================
---- STEP 5: Visualization 

W1 = opttheta[{{1,hiddenSize*visibleSize}}]:reshape(hiddenSize, visibleSize);
display_network(W1, 12); 
--print -djpeg weights.jpg   -- save the visualization to a file 
