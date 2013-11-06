--Concatenation of multiple tensors 
--(not sure if there is a more efficient way to do it)
function ncat(...) 
   local args = {...}
   local r = table.remove(args, 1)
   for k,x in ipairs(args) do
      r = torch.cat(r,x)
   end
   return r
end

function flatten(v)
    return v:reshape(#v:storage())
end
