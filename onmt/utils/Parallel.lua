--[[
  This file provides generic parallel class - allowing to run functions
  in different threads and on different GPU
]]--

local Parallel = {
  _pool = nil,
  count = 1,
  ompThreads = 1,
  gradBuffer = torch.Tensor()
}

local options = {
  {
    '-nparallel', 1,
    [[Number of parallel threads to run training on CPU.]]
  },
  {
    '-nompthreads', 44,
    [[Number of OpenMP threads used by each parallel thread.]]
  }
}

function Parallel.declareOpts(cmd)
  cmd:setCmdLineOptions(options, 'Parallel')
end

-- Synchronizes the current stream on dst device with src device. This is only
-- necessary if we are not on the default stream
local function waitForDevice(dst, src)
  if onmt.utils.Cuda.activated then
    local stream = cutorch.getStream()
      if stream ~= 0 then
        cutorch.streamWaitForMultiDevice(dst, stream, { [src] = {stream} })
      end
  end
end

function Parallel.getCounter()
  return Parallel._tds.AtomicCounter()
end

function Parallel.gmutexId()
  return Parallel._gmutex:id()
end

function Parallel.init(opt)
  Parallel.count = opt.nparallel
  Parallel.ompThreads = math.min(opt.nompthreads, math.floor(torch.getnumthreads()/Parallel.count))
  _G.logger:info("Running "..Parallel.count.." parallel threads")
  _G.logger:info("OMP num threads "..Parallel.ompThreads)

  if Parallel.count > 1 then
    Parallel._tds = require('tds')

    if Parallel.count > 1 then
      local globalLogger = _G.logger
      local globalProfiler = _G.profiler
      local threads = require('threads')
      local ompThreads = Parallel.ompThreads
      threads.Threads.serialization('threads.sharedserialize')
      Parallel._gmutex = threads.Mutex()
      Parallel._pool = threads.Threads(
        Parallel.count,
        function()
          require('sys')
          require('nngraph')
          require('onmt.init')
          _G.threads = require('threads')
        end,
        function(threadid)
          _G.logger = globalLogger
          _G.profiler = globalProfiler
          torch.setnumthreads(ompThreads)
          --torch.setnumthreads(1)
        end
      ) -- dedicate threads to GPUs
      Parallel._pool:specific(true)
    end

    --torch.setnumthreads(1)
    Parallel.usenccl = nil
  end
end

--[[ Launch function in parallel on different threads. ]]
function Parallel.launch(closure, endCallback)
  endCallback = endCallback or function() end
  for j = 1, Parallel.count do
    if Parallel._pool == nil then
      endCallback(closure(j))
    else
      Parallel._pool:addjob(j, function() return closure(j) end, endCallback)
    end
  end
  if Parallel._pool then
    Parallel._pool:synchronize()
  end
end

--[[ Accumulate the gradient parameters from the different parallel threads. ]]
function Parallel.accGradParams(gradParams, batches)
  if Parallel.count > 1 then
    for h = 1, #gradParams[1] do
      local inputs = { gradParams[1][h] }
      for j = 2, #batches do
        gradParams[1][h]:add(gradParams[j][h])
      end
    end
  end
end

-- [[ In async mode, sync the parameters from all replica to master replica. ]]
function Parallel.updateAndSync(masterParams, replicaGradParams, replicaParams, gradBuffer, masterGPU, gmutexId)
  -- Add a mutex to avoid competition while accessing shared buffer and while updating parameters.
  local mutex = _G.threads.Mutex(gmutexId)
  mutex:lock()
  local device = cutorch.getDevice()
  cutorch.setDevice(masterGPU)
  for h = 1, #replicaGradParams do
    waitForDevice(device, masterGPU)
    local remoteGrads = onmt.utils.Tensor.reuseTensor(gradBuffer, replicaGradParams[h]:size())
    remoteGrads:copy(replicaGradParams[h])
    waitForDevice(masterGPU, device)
    masterParams[h]:add(remoteGrads)
  end
  cutorch.setDevice(device)
  for h = 1, #replicaGradParams do
    replicaParams[h]:copy(masterParams[h])
    waitForDevice(device, masterGPU)
  end
  mutex:unlock()
end

--[[ Sync parameters from main model to different parallel threads. ]]
function Parallel.syncParams(params)
  if Parallel.count > 1 then
    if not Parallel.usenccl then
      for j = 2, Parallel.count do
        for h = 1, #params[1] do
          params[j][h]:copy(params[1][h])
        end
        waitForDevice(onmt.utils.Cuda.gpuIds[j], onmt.utils.Cuda.gpuIds[1])
      end
    else
      for h = 1, #params[1] do
        local inputs = { params[1][h] }
        for j = 2, Parallel.count do
          table.insert(inputs, params[j][h])
        end
        Parallel.usenccl.bcast(inputs, true, 1)
      end
    end
  end
end

return Parallel
