require('onmt.init')
require('tds')

local cmd = onmt.utils.ExtendedCmdLine.new('train.lua')

-- First argument define the model type: seq2seq/lm - default is seq2seq.
local modelType = cmd.getArgument(arg, '-model_type') or 'seq2seq'
local modelClass = onmt.ModelSelector(modelType)

-- Options declaration.
local options = {
  {
    '-data', '',
    [[Path to the data package `*-train.t7` generated by the preprocessing step.]],
    {
      valid = onmt.utils.ExtendedCmdLine.fileExists
    }
  }
}

cmd:setCmdLineOptions(options, 'Data')

onmt.data.SampledDataset.declareOpts(cmd)
onmt.Model.declareOpts(cmd)
modelClass.declareOpts(cmd)
onmt.train.Trainer.declareOpts(cmd)
onmt.utils.CrayonLogger.declareOpts(cmd)
onmt.utils.Cuda.declareOpts(cmd)
onmt.utils.Logger.declareOpts(cmd)
onmt.utils.Parallel.declareOpts(cmd)
onmt.utils.Dist.declareOpts(cmd)

cmd:text('')
cmd:text('Other options')
cmd:text('')

onmt.utils.Memory.declareOpts(cmd)
onmt.utils.Profiler.declareOpts(cmd)

cmd:option('-seed', 3435, [[Random seed.]], {valid=onmt.utils.ExtendedCmdLine.isUInt()})

local function loadDataset(filename)
  _G.logger:info('Loading data from \'%s\'...', filename)

  local dataset = torch.load(filename, 'binary', false)

  -- Keep backward compatibility.
  dataset.dataType = dataset.dataType or 'bitext'

  -- Check if data type is compatible with the target model.
  if not modelClass.dataType(dataset.dataType) then
    _G.logger:error('Data type `%s\' is incompatible with `%s\' models',
                    dataset.dataType, modelClass.modelName())
    os.exit(0)
  end

  return dataset
end

local function buildData(opt, dataset)
  local trainData
  if opt.sample > 0 then
     trainData = onmt.data.SampledDataset.new(opt, dataset.train.src, dataset.train.tgt)
  else
     trainData = onmt.data.Dataset.new(dataset.train.src, dataset.train.tgt)
  end
  local validData = onmt.data.Dataset.new(dataset.valid.src, dataset.valid.tgt)

  local nTrainBatch, batchUsage = trainData:setBatchSize(opt.max_batch_size, opt.uneven_batches)
  validData:setBatchSize(opt.max_batch_size, opt.uneven_batches)

  if dataset.dataType ~= 'monotext' then
    local srcVocSize
    local srcFeatSize = '-'
    if dataset.dicts.src then
      srcVocSize = dataset.dicts.src.words:size()
      srcFeatSize = #dataset.dicts.src.features
    else
      srcVocSize = '*'..dataset.dicts.srcInputSize
    end
    local tgtVocSize
    local tgtFeatSize = '-'
    if dataset.dicts.tgt then
      tgtVocSize = dataset.dicts.tgt.words:size()
      tgtFeatSize = #dataset.dicts.tgt.features
    else
      tgtVocSize = '*'..dataset.dicts.tgtInputSize
    end
    _G.logger:info(' * vocabulary size: source = %s; target = %s',
                   srcVocSize, tgtVocSize)
    _G.logger:info(' * additional features: source = %s; target = %s',
                   srcFeatSize, tgtFeatSize)
  else
    _G.logger:info(' * vocabulary size: %d', dataset.dicts.src.words:size())
    _G.logger:info(' * additional features: %d', #dataset.dicts.src.features)
  end
  _G.logger:info(' * maximum sequence length: source = %d; target = %d',
                 trainData.maxSourceLength, trainData.maxTargetLength)
  _G.logger:info(' * number of training sentences: %d', #trainData.src)
  _G.logger:info(' * number of batches: %d',  nTrainBatch)
  _G.logger:info('   - source sequence lengths: %s', opt.uneven_batches and 'variable' or 'equal')
  _G.logger:info('   - maximum size: %d', opt.max_batch_size)
  _G.logger:info('   - average size: %.2f', trainData:instanceCount() / nTrainBatch)
  _G.logger:info('   - capacity: %.2f%%', math.ceil(batchUsage * 1000) / 10)

  return trainData, validData
end

local function loadModel(opt, dicts)
  local checkpoint
  local paramChanges

  checkpoint, opt, paramChanges = onmt.train.Saver.loadCheckpoint(opt)

  cmd:logConfig(opt)

  local model = modelClass.load(opt, checkpoint.models, dicts)

  -- Change parameters dynamically.
  if not onmt.utils.Table.empty(paramChanges) then
    model:changeParameters(paramChanges)
  end

  return model, checkpoint.info
end

local function buildModel(opt, dicts)
  _G.logger:info('Building model...')
  return modelClass.new(opt, dicts)
end

local function main()
  local opt = cmd:parse(arg)

  torch.setdefaulttensortype("torch.FloatTensor")
  torch.manualSeed(opt.seed)

  -- Initialize global context.
  _G.logger = onmt.utils.Logger.new(opt.log_file, opt.disable_logs, opt.log_level)
  _G.crayon_logger = onmt.utils.CrayonLogger.new(opt)
  _G.profiler = onmt.utils.Profiler.new(false)

  onmt.utils.Cuda.init(opt)
  onmt.utils.Dist.init(opt)
  onmt.utils.Parallel.init(opt)

  _G.logger:info('Training ' .. modelClass.modelName() .. ' model...')

  -- Loading data package.
  local dataset = loadDataset(opt.data)

  -- Record data type in the options, and preprocessing options if present.
  opt.data_type = dataset.dataType
  opt.preprocess = dataset.opt

  -- Building training datasets.
  local trainData, validData = buildData(opt, dataset)

  -- Building the model.
  local model
  local trainStates

  if onmt.train.Saver.checkpointDefined(opt) then
    model, trainStates = loadModel(opt, dataset.dicts)
  else
    model = buildModel(opt, dataset.dicts)
  end

  onmt.utils.Cuda.convert(model)

  if opt.sample > 0 then
    trainData:checkModel(model)
  end

  -- Start training.
  local trainer = onmt.train.Trainer.new(opt, model, dataset.dicts, trainData:getBatch(1))
  trainer:train(trainData, validData, trainStates)

  _G.logger:shutDown()
  onmt.utils.Dist.finish(opt)
end

main()
