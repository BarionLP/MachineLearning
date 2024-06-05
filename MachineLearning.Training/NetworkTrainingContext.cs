using MachineLearning.Data.Entry;
using MachineLearning.Model;
using MachineLearning.Model.Layer;
using MachineLearning.Training.Evaluation;
using MachineLearning.Training.Optimization;
using MachineLearning.Training.Optimization.Layer;
using System.Collections.Immutable;

namespace MachineLearning.Training;

internal sealed class NetworkTrainingContext<TInput, TOutput>(INetwork<TInput, TOutput, RecordingLayer> network, TrainingConfig<TInput, TOutput> config, IOptimizer optimizer)
{
    internal INetwork<TInput, TOutput, RecordingLayer> Network = network;
    public TrainingConfig<TInput, TOutput> Config { get; } = config;
    internal ImmutableArray<ILayerOptimizer> LayerContexts = network.Layers.Select(optimizer.CreateLayerOptimizer).ToImmutableArray();
    internal ILayerOptimizer OutputLayerOptimizer => LayerContexts[^1];

    public void Train(IEnumerable<DataEntry<TInput, TOutput>> trainingBatch)
    {
        GradientCostReset();
        var dataCounter = 0;

        foreach(var dataPoint in trainingBatch)
        {
            Update(dataPoint);
            dataCounter++;
        }

        Apply(dataCounter);
    }

    public DataSetEvaluationResult TrainAndEvaluate(IEnumerable<DataEntry<TInput, TOutput>> trainingBatch)
    {
        GradientCostReset();
        int correctCounter = 0;
        double totalCost = 0;
        var dataCounter = 0;
        object _lock = new();

        foreach(var dataPoint in trainingBatch)
        {
           var weights = Update(dataPoint);
           var result = Network.Embedder.UnEmbed(weights)!;
           if(result.Equals(dataPoint.Expected))
           {
               correctCounter++;
           }
           dataCounter++;
           totalCost += Config.Optimizer.CostFunction.TotalCost(weights, Config.OutputResolver.Expected(dataPoint.Expected));
        }

        Apply(dataCounter);

        return new()
        {
            TotalCount = dataCounter,
            CorrectCount = correctCounter,
            TotalCost = totalCost,
        };
    }

    private void Apply(int dataCounter)
    {
        foreach(var layer in LayerContexts)
        {
            layer.Apply(dataCounter);
        }
    }

    private void GradientCostReset()
    {
        foreach(var layer in LayerContexts)
        {
            layer.GradientCostReset();
        }
    }

    public void FullReset()
    {
        foreach(var layer in LayerContexts)
        {
            layer.FullReset();
        }
    }

    private Vector Update(DataEntry<TInput, TOutput> data)
    {
        var result = Network.Forward(Network.Embedder.Embed(data.Input));

        var nodeValues = OutputLayerOptimizer.ComputeOutputLayerErrors(Config.OutputResolver.Expected(data.Expected));
        OutputLayerOptimizer.Update(nodeValues);


        for(int hiddenLayerIndex = LayerContexts.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--)
        {
            var hiddenLayer = LayerContexts[hiddenLayerIndex];
            nodeValues = hiddenLayer.ComputeHiddenLayerErrors(LayerContexts[hiddenLayerIndex + 1].Layer, nodeValues);
            hiddenLayer.Update(nodeValues);
        }

        return result;
    }
}