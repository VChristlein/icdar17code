function split(str, pat)
    local t = {}  -- NOTE: use {n = 0} in Lua-5.0
    local fpat = "(.-)" .. pat
    local last_end = 1
    local s, e, cap = str:find(fpat, 1)
    while s do
        if s ~= 1 or cap ~= "" then
            table.insert(t,cap)
        end
        last_end = e+1
        s, e, cap = str:find(fpat, last_end)
    end
    if last_end <= #str then
        cap = str:sub(last_end)
        table.insert(t, cap)
    end
    return t
end

require 'torch'
require 'xlua'
cmd = torch.CmdLine()
cmd:option('-d', 'Int', 'Data type')
cmd:option('-a', '10x10', 'Array Format')
cmd:option('-f', 'in.dat', 'Input file')
cmd:option('-o', 'out.dat',  'Output file')
params = cmd:parse(arg)

format = split(params.a, 'x')
for i=1,#format do
    format[i] = tonumber(format[i])
end

asize = torch.LongStorage(format)

f = torch.DiskFile(params.f)

f:binary()
f:quiet()

nitems = 1
for i=1,#format do nitems = nitems * format[i] end
if params.d == 'Int' then
    storage = f:readInt(nitems)
    tensor = torch.IntTensor(storage, 1, asize)
elseif params.d == 'Byte' then
    storage = f:readByte(nitems)
    tensor = torch.ByteTensor(storage, 1, asize)
elseif params.d == 'Float' then
    storage = f:readFloat(nitems)
    tensor = torch.FloatTensor(storage, 1, asize)
else
    error("Unknown file type")
end

f:close()

torch.save(params.o, tensor)
