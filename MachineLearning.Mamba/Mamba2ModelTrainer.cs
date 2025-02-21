using System.Collections.Immutable;
using System.Diagnostics;
using MachineLearning.Data;
using MachineLearning.Data.Entry;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training;
using MachineLearning.Training.Cost;
using MachineLearning.Training.Evaluation;
using MachineLearning.Training.Optimization;

namespace MachineLearning.Mamba;

public sealed class Mamba2ModelTrainer : ITrainer<Mamba2Model>
{
    public TrainingConfig Config { get; }
    public ITrainingSet TrainingSet { get; }
    public Mamba2Model Model { get; }
    public Optimizer Optimizer { get; }
    public ImmutableArray<ILayerOptimizer> LayerOptimizers { get; }
    public ILayerOptimizer OutputLayerOptimizer => LayerOptimizers[^1];

    public Mamba2ModelTrainer(Mamba2Model model, TrainingConfig config, ITrainingSet trainingSet)
    {
        Config = config;
        TrainingSet = trainingSet;
        Model = model;
        Optimizer = config.Optimizer;
        LayerOptimizers = [.. Model.Layers.Select(Optimizer.CreateLayerOptimizer)];
    }

    public void Train(CancellationToken? token = null)
    {
        Optimizer.Init();
        FullReset();
        var cachedEvaluation = DataSetEvaluationResult.ZERO;
        foreach (var (epochIndex, epoch) in TrainerHelper.GetEpochs(TrainingSet, Config.EpochCount).Index())
        {
            foreach (var (batchIndex, batch) in epoch.Index())
            {
                cachedEvaluation += TrainAndEvaluate(batch.Cast<TrainingData<Vector, Vector>>());
                if (Config.DumpBatchEvaluation && batchIndex % Config.DumpEvaluationAfterBatches == 0 || batchIndex + 1 == epoch.BatchCount && Config.DumpEpochEvaluation)
                {
                    Config.EvaluationCallback!.Invoke(new DataSetEvaluation { Context = GetContext(), Result = cachedEvaluation });
                    cachedEvaluation = DataSetEvaluationResult.ZERO;
                }
                Optimizer.OnBatchCompleted();

                if (token?.IsCancellationRequested is true)
                {
                    Optimizer.OnEpochCompleted();
                    return;
                }

                TrainingEvaluationContext GetContext() => new()
                {
                    CurrentBatch = batchIndex + 1,
                    MaxBatch = epoch.BatchCount,
                    CurrentEpoch = epochIndex + 1,
                    MaxEpoch = Config.EpochCount,
                    LearnRate = Config.Optimizer.LearningRate,
                };
            }

            Optimizer.OnEpochCompleted();
        }
    }

    public DataSetEvaluationResult TrainAndEvaluate(IEnumerable<TrainingData<Vector, Vector>> trainingBatch)
    {
        var timeStamp = Stopwatch.GetTimestamp();
        GradientCostReset();
        int correctCounter = 0;
        double totalCost = 0;
        var dataCounter = 0;

        if (Config.MultiThread)
        {
            var _lock = new Lock();
            Parallel.ForEach(trainingBatch, (data) =>
            {
                var weights = Update(data);

                lock (_lock)
                {
                    UpdateCounters(data, weights);
                }
            });
        }
        else
        {
            foreach (var data in trainingBatch)
            {
                var weights = Update(data);
                UpdateCounters(data, weights);
            }
        }

        void UpdateCounters(TrainingData<Vector, Vector> data, Vector weights)
        {
            dataCounter++;
            totalCost += Config.Optimizer.CostFunction.TotalCost(weights, data.ExpectedWeights);
        }

        Apply(dataCounter);

        return new()
        {
            TotalCount = dataCounter,
            CorrectCount = correctCounter,
            TotalCost = totalCost,
            TotalElapsedTime = Stopwatch.GetElapsedTime(timeStamp),
        };
    }

    private Vector Update(TrainingData<Vector, Vector> data)
    {
        var snapshots = Model.Layers.Select(LayerSnapshots.Get).Cast<Mamba2Layer.Snapshot>().ToImmutableArray();
        var result = Model.Process(data.InputValue, snapshots);

        var gradient = Optimizer.CostFunction.Derivative(result, data.ExpectedWeights);
        NumericsDebug.AssertValidNumbers(gradient);

        for (int layerIndex = LayerOptimizers.Length - 1; layerIndex >= 0; layerIndex--)
        {
            LayerOptimizers[layerIndex].Update(gradient, snapshots[layerIndex]);
            gradient = snapshots[layerIndex].GradientInput;
            NumericsDebug.AssertValidNumbers(gradient);
        }

        foreach (var (layer, snapshot) in Model.Layers.Zip(snapshots))
        {
            LayerSnapshots.Return(layer, snapshot);
        }

        return result;
    }

    private void Apply(int dataCounter) => LayerOptimizers.Consume(layer => layer.Apply(dataCounter));
    private void GradientCostReset() => LayerOptimizers.Consume(layer => layer.GradientCostReset());
    private void FullReset() => LayerOptimizers.Consume(layer => layer.FullReset());
}

public sealed class EmbeddedMamba2ModelTrainer : ITrainer<EmbeddedMamba2Model>
{
    public TrainingConfig Config { get; }
    public ITrainingSet TrainingSet { get; }
    public EmbeddedMamba2Model Model { get; }
    public Optimizer Optimizer { get; }
    public ILayerOptimizer InputLayerOptimizer { get; }
    public ImmutableArray<ILayerOptimizer> LayerOptimizers { get; }
    public ILayerOptimizer OutputLayerOptimizer { get; }

    public EmbeddedMamba2ModelTrainer(EmbeddedMamba2Model model, TrainingConfig config, ITrainingSet trainingSet)
    {
        Config = config;
        TrainingSet = trainingSet;
        Model = model;
        Optimizer = config.Optimizer;
        InputLayerOptimizer = Optimizer.CreateLayerOptimizer(Model.InputLayer);
        LayerOptimizers = [.. Model.HiddenLayers.Select(Optimizer.CreateLayerOptimizer)];
        OutputLayerOptimizer = Optimizer.CreateLayerOptimizer(Model.OutputLayer);
    }

    public void Train(CancellationToken? token = null)
    {
        Optimizer.Init();
        FullReset();
        var cachedEvaluation = DataSetEvaluationResult.ZERO;
        foreach (var (epochIndex, epoch) in TrainerHelper.GetEpochs(TrainingSet, Config.EpochCount).Index())
        {
            foreach (var (batchIndex, batch) in epoch.Index())
            {
                cachedEvaluation += TrainAndEvaluate(batch.Cast<TrainingData<int[], int>>());
                if (Config.DumpBatchEvaluation && batchIndex % Config.DumpEvaluationAfterBatches == 0 || batchIndex + 1 == epoch.BatchCount && Config.DumpEpochEvaluation)
                {
                    Config.EvaluationCallback!.Invoke(new DataSetEvaluation { Context = GetContext(), Result = cachedEvaluation });
                    cachedEvaluation = DataSetEvaluationResult.ZERO;
                }
                Optimizer.OnBatchCompleted();

                if (token?.IsCancellationRequested is true)
                {
                    Optimizer.OnEpochCompleted();
                    return;
                }

                TrainingEvaluationContext GetContext() => new()
                {
                    CurrentBatch = batchIndex + 1,
                    MaxBatch = epoch.BatchCount,
                    CurrentEpoch = epochIndex + 1,
                    MaxEpoch = Config.EpochCount,
                    LearnRate = Config.Optimizer.LearningRate,
                };
            }

            Optimizer.OnEpochCompleted();
        }
    }

    public DataSetEvaluationResult TrainAndEvaluate(IEnumerable<TrainingData<int[], int>> trainingBatch)
    {
        var timeStamp = Stopwatch.GetTimestamp();
        GradientCostReset();
        int correctCounter = 0;
        double totalCost = 0;
        var dataCounter = 0;

        if (Config.MultiThread)
        {
            var _lock = new Lock();
            Parallel.ForEach(trainingBatch, (data) =>
            {
                var (weights, result) = Update(data);

                lock (_lock)
                {
                    UpdateCounters(data, weights, result);
                }
            });
        }
        else
        {
            foreach (var data in trainingBatch)
            {
                var (weights, result) = Update(data);
                UpdateCounters(data, weights, result);
            }
        }

        void UpdateCounters(TrainingData<int[], int> data, Vector weights, int result)
        {
            if (result == data.ExpectedValue)
            {
                correctCounter++;
            }
            dataCounter++;
            totalCost += Config.Optimizer.CostFunction.TotalCost(weights, data.ExpectedWeights);
        }

        Apply(dataCounter);

        return new()
        {
            TotalCount = dataCounter,
            CorrectCount = correctCounter,
            TotalCost = totalCost,
            TotalElapsedTime = Stopwatch.GetElapsedTime(timeStamp),
        };
    }

    private (Vector, int) Update(TrainingData<int[], int> data)
    {
        var inputSnapshot = LayerSnapshots.Get(Model.InputLayer);
        var snapshots = Model.HiddenLayers.Select(LayerSnapshots.Get).Cast<EmbeddedMamba2Layer.Snapshot>().ToImmutableArray();
        var outputSnapshot = (UnEmbeddingLayer.Snapshot)LayerSnapshots.Get(Model.OutputLayer);

        var (weights, result) = Model.Process(data.InputValue, [inputSnapshot, .. snapshots, outputSnapshot]);

        var gradient = Optimizer.CostFunction.Derivative(weights, data.ExpectedWeights);
        NumericsDebug.AssertValidNumbers(gradient);

        OutputLayerOptimizer.Update(gradient, outputSnapshot);
        gradient = outputSnapshot.InputGradient.Storage;

        for (int layerIndex = LayerOptimizers.Length - 1; layerIndex >= 0; layerIndex--)
        {
            LayerOptimizers[layerIndex].Update(gradient, snapshots[layerIndex]);
            gradient = snapshots[layerIndex].GradientInput.Storage;
            NumericsDebug.AssertValidNumbers(gradient);
        }

        InputLayerOptimizer.Update(gradient, inputSnapshot);

        LayerSnapshots.Return(Model.InputLayer, inputSnapshot);
        LayerSnapshots.Return(Model.OutputLayer, outputSnapshot);
        foreach (var (layer, snapshot) in Model.HiddenLayers.Zip(snapshots))
        {
            LayerSnapshots.Return(layer, snapshot);
        }

        return (weights, result);
    }

    private void Apply(int dataCounter)
    {
        InputLayerOptimizer.Apply(dataCounter);
        LayerOptimizers.Consume(layer => layer.Apply(dataCounter));
        OutputLayerOptimizer.Apply(dataCounter);
    }
    private void GradientCostReset()
    {
        InputLayerOptimizer.GradientCostReset();
        LayerOptimizers.Consume(layer => layer.GradientCostReset());
        OutputLayerOptimizer.GradientCostReset();
    }
    private void FullReset()
    {
        InputLayerOptimizer.FullReset();
        LayerOptimizers.Consume(layer => layer.FullReset());
        OutputLayerOptimizer.FullReset();
    }
}
