using System.Collections.Immutable;
using MachineLearning.Data.Entry;
using MachineLearning.Model;
using MachineLearning.Model.Layer.Snapshot;
using MachineLearning.Training.Evaluation;
using MachineLearning.Training.Optimization;

namespace MachineLearning.Training;

public sealed class GenericModelTrainer<TInput, TOutput>
{
    public TrainingConfig<TInput, TOutput> Config { get; }
    public IEmbeddedModel<TInput, TOutput> Model { get; }
    public IGenericOptimizer Optimizer { get; }
    public ImmutableArray<ILayerOptimizer> LayerOptimizers { get; }
    public ILayerOptimizer OutputLayerOptimizer => LayerOptimizers[LastUsableLayerIndex];
    private static readonly Index LastUsableLayerIndex = ^2;

    public GenericModelTrainer(IEmbeddedModel<TInput, TOutput> model, TrainingConfig<TInput, TOutput> config)
    {
        Config = config;
        Model = model;
        Optimizer = config.Optimizer;
        LayerOptimizers = Model.Layers.Select(Optimizer.CreateLayerOptimizer).ToImmutableArray();
    }

    public void TrainConsole(bool cancelable = true)
    {
        using var cts = new CancellationTokenSource();
        if (cancelable)
        {
            Task.Run(async () =>
            {
                while (!cts.IsCancellationRequested)
                {
                    if (Console.KeyAvailable && Console.ReadKey(intercept: true).Key == ConsoleKey.C)
                    {
                        Console.WriteLine("Canceling...");
                        cts.Cancel();
                        break;
                    }
                    await Task.Delay(500);
                }
            });
        }

        Console.WriteLine($"Training {Model}");
        Console.WriteLine(Config.ToString());
        Console.WriteLine("Starting Training...");
        Train(cts.Token);
        cts.Cancel();
        Console.WriteLine("Training Done!");
    }

    public void Train(CancellationToken? token = null)
    {
        Optimizer.Init();
        FullReset();
        var cachedEvaluation = DataSetEvaluationResult.ZERO;
        foreach (var epochIndex in ..Config.EpochCount)
        {
            var epoch = Config.GetEpoch();
            var batchCount = 0;

            foreach (var batch in epoch)
            {
                cachedEvaluation += TrainAndEvaluate(batch, multithread: true);
                if ((Config.DumpBatchEvaluation && batchCount % Config.DumpEvaluationAfterBatches == 0) || (batchCount + 1 == epoch.BatchCount && Config.DumpEpochEvaluation))
                {
                    Config.EvaluationCallback!.Invoke(new DataSetEvaluation { Context = GetContext(), Result = cachedEvaluation });
                    cachedEvaluation = DataSetEvaluationResult.ZERO;
                }
                batchCount++;
                Optimizer.OnBatchCompleted();

                if (token?.IsCancellationRequested is true)
                {
                    Optimizer.OnEpochCompleted();
                    return;
                }
            }

            Optimizer.OnEpochCompleted();

            TrainingEvaluationContext GetContext() => new()
            {
                CurrentBatch = batchCount + 1,
                MaxBatch = epoch.BatchCount,
                CurrentEpoch = epochIndex + 1,
                MaxEpoch = Config.EpochCount,
                LearnRate = Config.Optimizer.LearningRate,
            };
        }
    }

    public DataSetEvaluationResult TrainAndEvaluate(IEnumerable<DataEntry<TInput, TOutput>> trainingBatch, bool multithread)
    {
        var sw = Stopwatch.StartNew();
        GradientCostReset();
        int correctCounter = 0;
        double totalCost = 0;
        var dataCounter = 0;

        if (multithread)
        {
            var _lock = new Lock();
            Parallel.ForEach(trainingBatch, (dataPoint) =>
            {
                var (result, _, weights) = Update(dataPoint);

                lock (_lock)
                {
                    if (result!.Equals(dataPoint.Expected))
                    {
                        correctCounter++;
                    }
                    dataCounter++;
                    totalCost += Config.Optimizer.CostFunction.TotalCost(weights, Config.OutputResolver.Expected(dataPoint.Expected));
                }
            });
        }
        else
        {
            foreach (var dataPoint in trainingBatch)
            {
                var (result, _, weights) = Update(dataPoint);
                if (result!.Equals(dataPoint.Expected))
                {
                    correctCounter++;
                }
                dataCounter++;
                totalCost += Config.Optimizer.CostFunction.TotalCost(weights, Config.OutputResolver.Expected(dataPoint.Expected));
            }
        }


        Apply(dataCounter);

        return new()
        {
            TotalCount = dataCounter,
            CorrectCount = correctCounter,
            TotalCost = totalCost,
            TotalElapsedTime = sw.Elapsed,
        };
    }

    private (TOutput output, int index, Vector weights) Update(DataEntry<TInput, TOutput> data)
    {
        var snapshots = Model.Layers.Select(LayerSnapshots.Get).ToImmutableArray();
        var result = Model.Forward(data.Input, snapshots);

        //last layer gets skipped right now because it never contains weights (unembedding layer). i will change this in the future to allow trained unembedding
        var nodeValues = LayerBackPropagation.ComputeOutputLayerErrors(OutputLayerOptimizer.Layer, OutputLayerOptimizer.CostFunction, Config.OutputResolver.Expected(data.Expected), snapshots[LastUsableLayerIndex]);
        OutputLayerOptimizer.Update(nodeValues, snapshots[LastUsableLayerIndex]);


        for (int hiddenLayerIndex = LayerOptimizers.Length - 3; hiddenLayerIndex >= 0; hiddenLayerIndex--)
        {
            var hiddenLayer = LayerOptimizers[hiddenLayerIndex];
            nodeValues = LayerBackPropagation.ComputeHiddenLayerErrors(hiddenLayer.Layer, LayerOptimizers[hiddenLayerIndex + 1].Layer, nodeValues, snapshots[hiddenLayerIndex]);
            NumericsDebug.RequireValidNumbers(nodeValues);
            hiddenLayer.Update(nodeValues, snapshots[hiddenLayerIndex]);
        }

        //TODO: verify zip performance
        foreach (var (layer, snapshot) in Model.Layers.Zip(snapshots))
        {
            LayerSnapshots.Return(layer, snapshot);
        }

        return result;
    }

    private void Apply(int dataCounter) => LayerOptimizers.ForEach(layer => layer.Apply(dataCounter));
    private void GradientCostReset() => LayerOptimizers.ForEach(layer => layer.GradientCostReset());
    private void FullReset() => LayerOptimizers.ForEach(layer => layer.FullReset());
}