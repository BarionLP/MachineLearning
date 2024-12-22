using System.Collections.Immutable;
using MachineLearning.Data;
using MachineLearning.Data.Entry;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training.Evaluation;
using MachineLearning.Training.Optimization;

namespace MachineLearning.Training;

public sealed class EmbeddedModelTrainer<TIn, TOut> : ITrainer<EmbeddedModel<TIn, TOut>>
{
    public TrainingConfig Config { get; }
    public ITrainingSet TrainingSet { get; }
    public EmbeddedModel<TIn, TOut> Model { get; }
    public Optimizer Optimizer { get; }
    public ImmutableArray<ILayerOptimizer> LayerOptimizers { get; }
    public ILayerOptimizer OutputLayerOptimizer => LayerOptimizers[^1];

    public EmbeddedModelTrainer(EmbeddedModel<TIn, TOut> model, TrainingConfig config, ITrainingSet trainingSet)
    {
        Config = config;
        TrainingSet = trainingSet;
        Model = model;
        Optimizer = config.Optimizer;
        LayerOptimizers = Model.InnerModel.Layers.Select(Optimizer.CreateLayerOptimizer).ToImmutableArray();
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
                cachedEvaluation += TrainAndEvaluate(batch.OfType<TrainingData<TIn, TOut>>());
                if ((Config.DumpBatchEvaluation && batchIndex % Config.DumpEvaluationAfterBatches == 0) || (batchIndex + 1 == epoch.BatchCount && Config.DumpEpochEvaluation))
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

    public DataSetEvaluationResult TrainAndEvaluate(IEnumerable<TrainingData<TIn, TOut>> trainingBatch)
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

        void UpdateCounters(TrainingData<TIn, TOut> data, Vector weights)
        {
            if (Model.OutputLayer.Process(weights).output!.Equals(data.ExpectedValue))
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

    private Vector Update(TrainingData<TIn, TOut> data)
    {
        var snapshots = Model.InnerModel.Layers.Select(LayerSnapshots.Get).OfType<LayerSnapshots.Simple>().ToImmutableArray();
        var inputWeights = Model.InputLayer.Process(data.InputValue);
        var result = Model.InnerModel.Process(inputWeights, snapshots);

        var nodeValues = LayerBackPropagation.ComputeOutputLayerErrors(Model.InnerModel.Layers[^1], OutputLayerOptimizer.CostFunction, data.ExpectedWeights, snapshots[^1]);
        NumericsDebug.AssertValidNumbers(nodeValues);
        OutputLayerOptimizer.Update(nodeValues, snapshots[^1]);


        for (int hiddenLayerIndex = LayerOptimizers.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--)
        {
            var hiddenLayer = LayerOptimizers[hiddenLayerIndex];
            nodeValues = LayerBackPropagation.ComputeHiddenLayerErrors(Model.InnerModel.Layers[hiddenLayerIndex], Model.InnerModel.Layers[hiddenLayerIndex + 1], nodeValues, snapshots[hiddenLayerIndex]);
            NumericsDebug.AssertValidNumbers(nodeValues);
            hiddenLayer.Update(nodeValues, snapshots[hiddenLayerIndex]);
        }

        //TODO: verify zip performance
        foreach (var (layer, snapshot) in Model.InnerModel.Layers.Zip(snapshots))
        {
            LayerSnapshots.Return(layer, snapshot);
        }

        return result;
    }

    private void Apply(int dataCounter) => LayerOptimizers.Consume(layer => layer.Apply(dataCounter));
    private void GradientCostReset() => LayerOptimizers.Consume(layer => layer.GradientCostReset());
    private void FullReset() => LayerOptimizers.Consume(layer => layer.FullReset());
}

public interface ITrainer<TModel>
{
    public TModel Model { get; }
    public TrainingConfig Config { get; }
    public ITrainingSet TrainingSet { get; }
    void Train(CancellationToken? token = null);
}
