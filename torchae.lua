require("io")
require("string")

function table.reverse ( tab )
    local size = #tab
    local newTable = {}

    for i,v in ipairs ( tab ) do
        newTable[size-i+1] = v
    end

    return newTable
end

function load_mat(filename)
    local rv=nil
    local rv_s=nil
    i=0
    file = io.open(filename)
    for line in file:lines() do
        if not string.match(line, "^#") then
            --check if we haven't created yet the output tensor
            --(because we didn't know the dimensions yet)
            if not rv then
                --read the dimensions 
                dims = {}
                for dim in string.gmatch(line, "%d+") do
                    dims[#dims + 1] = tonumber(dim)
                end
                --create a tensor with the specified dimensions
                rv = torch.Tensor(unpack(table.reverse(dims)))
                --we will use the storage (actual in-memory array) to
                --store the data
                rv_s = rv:storage()
            elseif string.len(line) > 0 then
                i = i + 1
                if not pcall(function()
                rv_s[i] = tonumber(line)
                end) then 
                    print ("error processing line: "..line)
                end
            end
        end
    end
    return rv
end
