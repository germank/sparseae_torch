function computeNumericalGradient(J, theta)
    -- numgrad = computeNumericalGradient(J, theta)
    -- theta: a vector of parameters
    -- J: a function that outputs a real-number. Calling y = J(theta) will return the
    -- function value at theta. 
      
    -- Initialize numgrad with zeros
    local numgrad = torch.zeros(theta:size());

    ---- ---------- YOUR CODE HERE --------------------------------------
    -- Instructions: 
    -- Implement numerical gradient checking, and return the result in numgrad.  
    -- (See Section 2.3 of the lecture notes.)
    -- You should write code so that numgrad(i) is (the numerical approximation to) the 
    -- partial derivative of J with respect to the i-th input argument, evaluated at theta.  
    -- I.e., numgrad(i) should be the (approximately) the partial derivative of J with 
    -- respect to theta(i).
    --                
    -- Hint: You will probably want to compute the elements of numgrad one at a time. 

    local e = 0.0001
   
    local e_i = torch.zeros(theta:size())
    for i=1,theta:size()[1] do
        print (i)
        e_i[i] = e
      numgrad[i] = (J(theta + e_i) - J(theta - e_i)) / (2*e)
        e_i[i] = 0
    end





    return numgrad
    ---- ---------------------------------------------------------------
end
