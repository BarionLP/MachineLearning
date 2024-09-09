using MachineLearning.Data.Entry;
using MachineLearning.Model;
using MachineLearning.Model.Embedding;
using MachineLearning.Training.Evaluation;
using MachineLearning.Training.Optimization;
using System.Collections.Immutable;
using System.Diagnostics;

namespace MachineLearning.Training;

public sealed class ModelTrainingContext<TInput, TOutput>(EmbeddedModel<TInput, TOutput> model, TrainingConfig<TInput, TOutput> config, IOptimizer optimizer)
{
    internal SimpleModel Model = model.InternalModel;
    internal IEmbedder<TInput, TOutput> Embedder = model.Embedder;
    public TrainingConfig<TInput, TOutput> Config { get; } = config;
    internal ImmutableArray<ILayerOptimizer> LayerOptimizers = model.InternalModel.Layers.Select(optimizer.CreateLayerOptimizer).ToImmutableArray();
    internal ILayerOptimizer OutputLayerOptimizer => LayerOptimizers[^1];

    public DataSetEvaluationResult TrainAndEvaluate(IEnumerable<DataEntry<TInput, TOutput>> trainingBatch, bool multithread)
    {
        var sw = Stopwatch.StartNew();
        GradientCostReset();
        int correctCounter = 0;
        double totalCost = 0;
        var dataCounter = 0;

        if (multithread)
        {
            object _lock = new();
            Parallel.ForEach(trainingBatch, (dataPoint) =>
            {
                var weights = Update(dataPoint);
                var (result, _) = Embedder.Unembed(weights);

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
                var weights = Update(dataPoint);
                var (result, _) = Embedder.Unembed(weights);
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

    private void Apply(int dataCounter)
    {
        foreach(var layer in LayerOptimizers)
        {
            layer.Apply(dataCounter);
        }
    }

    private void GradientCostReset()
    {
        foreach(var layer in LayerOptimizers)
        {
            layer.GradientCostReset();
        }
    }

    public void FullReset()
    {
        foreach(var layer in LayerOptimizers)
        {
            layer.FullReset();
        }
    }

    private Vector Update(DataEntry<TInput, TOutput> data)
    {
        var snapshots = Model.Layers.Select(LayerSnapshot.Get).ToImmutableArray();
        var result = Model.Forward(Embedder.Embed(data.Input), snapshots);

        var nodeValues = OutputLayerOptimizer.ComputeOutputLayerErrors(Config.OutputResolver.Expected(data.Expected), snapshots[^1]);
        OutputLayerOptimizer.Update(nodeValues, snapshots[^1]);


        for(int hiddenLayerIndex = LayerOptimizers.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--)
        {
            var hiddenLayer = LayerOptimizers[hiddenLayerIndex];
            nodeValues = hiddenLayer.ComputeHiddenLayerErrors(LayerOptimizers[hiddenLayerIndex + 1].Layer, nodeValues, snapshots[hiddenLayerIndex]);
            hiddenLayer.Update(nodeValues, snapshots[hiddenLayerIndex]);
        }


        //TODO: verify zip performance
        foreach(var (layer, snapshot) in Model.Layers.Zip(snapshots)){
            LayerSnapshot.Return(layer, snapshot);
        }

        return result;
    }
}