--[[
  This file provides generic dist class - allowing to run functions
  in different MPI ranks
]]--

local Dist = {
  usegpu = false,
  rank = 1,
  size = 1,
}

local options = {
  {
    '-use_dist', true,
    [[Use distributed training via TorchMPI]]
  },
  {
    '-debug_print', true,
    [[Use debug print mode with MPI info]]
  }
}

function Dist.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Dist')
end

local function initPrint(opt)
  if opt.debug_print then
    -- overwrite print
    old_print = print
    print = function(...)
      io.write(string.format('[%d/%d] ', Dist.rank, Dist.size))
      old_print(...)
    end
  else
    -- print only on host node
    print = function() end
  end
end

function Dist.init(opt)
  if opt.use_dist then
    Dist.mpi = require('torchmpi')
    Dist.mpi.start(Dist.usegpu)

    -- indexing from 1
    Dist.rank = Dist.mpi.rank() + 1
    Dist.size = Dist.mpi.size()

    initPrint(opt)
  end
end

--[[ Accumulate the gradient parameters from the different nodes. ]]
function Dist.accGradParams(gradParams)
  for h = 1, #gradParams[1] do
    Dist.mpi.allreduceTensor(gradParams[1][h])
  end
end

--[[ Sync parameters from main model to different nodes. ]]
function Dist.syncParams(params)
  for h = 1, #params[1] do
    Dist.mpi.broadcastTensor(0, params[1][h])
  end
end

--[[ All reduce across different nodes]]
function Dist.allreduce(scalar)
  scalarTensor = torch.Tensor({scalar})
  Dist.mpi.allreduceTensor(scalarTensor)
  return scalarTensor[1]
end

function Dist.finish()
  Dist.mpi.stop()
end 

return Dist
