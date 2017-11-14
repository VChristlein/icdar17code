require 'torch'

local binaryIO = {}

local BinaryIO = {}

function BinaryIO:readHeader()
    self.dchar = string.char(self.f:readChar())
    self.rows = self.f:readInt()
    self.cols = self.f:readInt()
    self.channels = self.f:readInt()
--    print(self.dtype..' '..self.rows..' '..self.cols)
end

function BinaryIO:writeHeader(dchar, rows, cols, channels)
    self.dchar = dchar
    self.rows = rows
    self.cols = cols
    self.channels = channels
    self.f:writeChar(self.dtype)
    self.f:writeInt(rows)
    self.f:writeInt(cols)
    self.f:writeInt(channels)
end

function BinaryIO:writeTensor(tensor)
    if self.dchar == 'u' then
        self.f:writeByte(tensor:storage())
    elseif self.dchar == 'i' then
        self.f:writeInt(tensor:storage())
    elseif self.dchar == 'f' then
        self.f:writeFloat(tensor:storage())
    elseif self.dchar == 'd' then
        self.f:writeDouble(tensor:storage())
    end
end

function BinaryIO:readTensor()
    local storage
    local tensor
    local sqrtc = math.sqrt(self.cols)
    if self.dchar == 'u' then
        storage = self.f:readByte(self.rows*self.cols*self.channels)
        tensor = torch.ByteTensor(storage, 1, torch.LongStorage{self.rows, self.channels, sqrtc, sqrtc})
    elseif self.dchar == 'i' then
        storage = self.f:readInt(self.rows*self.cols*self.channels)
        tensor = torch.IntTensor(storage, 1, torch.LongStorage{self.rows, self.channels, sqrtc, sqrtc})
    elseif self.dchar == 'f' then
        storage = self.f:readFloat(self.rows*self.cols*self.channels)
        tensor = torch.FloatTensor(storage, 1, torch.LongStorage{self.rows, self.channels, sqrtc, sqrtc})
    elseif self.dchar == 'd' then
        storage = self.f:readDouble(self.rows*self.cols*self.channels)
        tensor = torch.DoubleTensor(storage, 1, torch.LongStorage{self.rows, self.channels, sqrtc, sqrtc})
    end
    return tensor
end

function BinaryIO:open(fname, fmode)
    self.f = torch.DiskFile(fname, fmode)
    self.f:binary()
    if fmode == 'r' then
        self:readHeader()
    end
end

function BinaryIO:close()
    self.f:close()
    self.rows = nil
    self.cols = nil
    self.dchar = nil
    self.channels = nil
    self.f = nil
end

function binaryIO.new(fname, fmode)
    local self = {}
    setmetatable(self, { __index = BinaryIO })
    self:open(fname, fmode)
    return self
end

return binaryIO
