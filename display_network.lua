function display_network(A, opt_normalize, opt_graycolor, cols, opt_colmajor)
-- This function visualizes filters in matrix A. Each column of A is a
-- filter. We will reshape each column into a square image and visualizes
-- on each cell of the visualization panel. 
-- All other parameters are optional, usually you do not need to worry
-- about it.
-- opt_normalize: whether we need to normalize the filter so that all of
-- them can have similar contrast. Default value is true.
-- opt_graycolor: whether we use gray as the heat map. Default is true.
-- cols: how many columns are there in the display. Default value is the
-- squareroot of the number of columns in A.
-- opt_colmajor: you can switch convention to row major for A. In that
-- case, each row of A is a filter. Default value is false.

    local opt_normalize = opt_normalize or true
    local opt_graycolor = opt_graycolor or true
    local opt_colmajor = opt_colmajor or false

    -- rescale
    local A = A - torch.mean(A)

    local colormap=nil
    if opt_graycolor then
        colormap = 'gray'
    else
        colormap = 'color'
    end

    -- compute rows, cols
    local M,L=A:size()[1],A:size()[2]
    local sz=math.sqrt(L)
    local buf=1
    local m,n
    if not cols then
        if math.floor(math.sqrt(M))^2 ~= M then
            n=math.ceil(math.sqrt(M))
            while M % n~=0 and n<1.2*math.sqrt(M) do n=n+1 end
            m=math.ceil(M/n)
        else
            local n=math.sqrt(M)
            m=n
        end
    else
        n = cols
        m = math.ceil(M/n)
    end

    array=-torch.Tensor(buf+m*(sz+buf),buf+n*(sz+buf)):fill(1)

    if not opt_graycolor then
        array = array * 0.1
    end


    if not opt_colmajor then
        local k=1
        for i=1,m do
            for j=1,n do
                if k<=M then 
                    local clim=torch.max(torch.abs(A[{k,{}}]))
                    local xoff=buf+(i-1)*(sz+buf)
                    local yoff=buf+(j-1)*(sz+buf)
                    if opt_normalize then
                        array[{{1+xoff,sz+xoff},{1+yoff,sz+yoff}}]=A[{k,{}}]:resize(sz,sz)/clim
                    else
                        array[{{1+xoff,sz+xoff},{1+yoff,sz+yoff}}]=A[{k{}}]:resize(sz,sz)/ torch.max(torch.abs(A))
                    end
                    k=k+1
                end
            end
        end
    else
        local k=1
        for j=1,n do
            for i=1,m do
                if k<=M then 
                    local clim=torch.max(torch.abs(A[{k,{}}]))
                    local xoff=buf+(i-1)*(sz+buf)
                    local yoff=buf+(j-1)*(sz+buf)
                    if opt_normalize then
                        array[{{1+xoff,sz+yoff},{1+yoff,sz+yoff}}]=A[{k,{}}]:resize(sz,sz)/clim
                    else
                        arrayarray[{{1+xoff,sz+xoff},{1+yoff,sz+yoff}}]=A[{k,{}}]:resize(sz,sz)
                    end
                    k=k+1
                end
            end
        end
    end
   
    require 'image'
    if opt_graycolor then
        image.display(array)
    else
        image.display(array)
    end

    return h, array
end
