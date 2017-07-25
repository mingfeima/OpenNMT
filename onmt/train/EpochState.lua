--[[ Class for managing the training process by logging and storing
  the state of the current epoch.
]]
local EpochState = torch.class('EpochState')

--[[ Initialize for epoch `epoch`]]
function EpochState:__init(epoch, startIterations, numIterations, learningRate, optimStatus)
  self.epoch = epoch
  self.iterations = startIterations - 1
  self.numIterations = numIterations
  self.learningRate = learningRate
  self.optimStatus = optimStatus

  self.globalTimer = torch.Timer()

  self:reset()
end

function EpochState:reset()
  self.trainLoss = 0
  self.sourceWords = 0
  self.targetWords = 0

  self.timer = torch.Timer()
end

--[[ Update training status. Takes `batch` (described in data.lua) and last loss.]]
function EpochState:update(model, batch, loss)
  self.iterations = self.iterations + 1
  self.trainLoss = self.trainLoss + loss
  self.sourceWords = self.sourceWords + model:getInputLabelsCount(batch)
  self.targetWords = self.targetWords + model:getOutputLabelsCount(batch)
end

--[[ Log to status stdout. ]]
function EpochState:log(iteration)
  -- Synchronize across ranks
  local _trainLoss = onmt.utils.Dist.allreduce(self.trainLoss)
  local _targetWords = onmt.utils.Dist.allreduce(self.targetWords)
  local _sourceWords = onmt.utils.Dist.allreduce(self.sourceWords)
  
  local ppl = math.exp(_trainLoss / _targetWords)
  local tokpersec = _sourceWords / self.timer:time().real
  _G.logger:info('Epoch %d ; Iteration %d/%d ; %s ; Source tokens/s %d ; Perplexity %.2f ; Time %.3f sec',
                  self.epoch,
                  iteration or self.iterations, self.numIterations,
                  self.optimStatus,
                  tokpersec,
                  ppl,
                  self.timer:time().real)
  if _G.crayon_logger.on == true then
     _G.crayon_logger.exp:add_scalar_value("learning_rate", self.learningRate)
     _G.crayon_logger.exp:add_scalar_value("perplexity", ppl)
     _G.crayon_logger.exp:add_scalar_value("token_per_sec", tokpersec)
  end
  self:reset()
end

function EpochState:getTime()
  return self.globalTimer:time().real
end

return EpochState
