function sample_images(IMAGES)
   -- Returns 10000 patches for training
   
   local patchsize = 8  -- we'll use 8x8 patches 
   local numpatches = 10000
   
   -- Initialize patches with zeros.  Your code will fill in this matrix--one
   -- row per patch, 10000 columns. 
   local patches = torch.Tensor(numpatches,patchsize*patchsize):zero()
   
   ---- ---------- YOUR CODE HERE --------------------------------------
   --  Instructions: Fill in the variable called "patches" using data 
   --  from IMAGES.  
   --  
   --  IMAGES is a 3D array containing 10 images
   --  For instance, IMAGES[{6,{},{}}] is a 512x512 array containing the 6th image,
   --  and you can type "gnuplot.imagesc(IMAGES[6], 'gray');" to visualize
   --  it. (The contrast on these images look a bit off because they have
   --  been preprocessed using using "whitening."  See the lecture notes for
   --  more details.) As a second example, IMAGES[{{1},{21,30},{21,30}}][1] is an image
   --  patch corresponding to the pixels in the block (21,21) to (30,30) of
   --  Image 1
   local imgsize = IMAGES:size(2)
   local row = 1
   local col = 1
   local img = 1
   for i=1,numpatches do
      patches[i] = 
         IMAGES[{{img},{row,row+patchsize-1},{col,col+patchsize-1}}][1]
      col = col + patchsize
      if col >= imgsize then
         col = 1
         row = row + patchsize
      end
      if row >= imgsize then
         row = 1
         img = img + 1
      end
   end


   
   ---- ---------------------------------------------------------------
   -- For the autoencoder to work well we need to normalize the data
   -- Specifically, since the output of the network is bounded between [0,1]
   -- (due to the sigmoid activation function), we have to make sure 
   -- the range of pixel values is also bounded between [0,1]
   patches = normalizeData(patches);
   
   return patches
end


---- ---------------------------------------------------------------
function normalizeData(patches)

-- Squash data to [0.1, 0.9] since we use sigmoid as the activation
-- function in the output layer

   -- Remove DC (mean of images). 
   local patches_means = torch.mean(patches,2)
   for i=1,patches:size(1) do
      patches[i]:apply(function(x) return x + patches_means[i][1] end)
   end
   -- Truncate to +/-3 standard deviations and scale to -1 to 1
   local pstd = 3 * torch.std(patches);
   patches = patches:apply(function(x) return math.max(math.min(x, pstd), -pstd) / pstd end);

-- Rescale from [-1,1] to [0.1,0.9]
   patches = (patches + 1) * 0.4 + 0.1;
   return patches
end
