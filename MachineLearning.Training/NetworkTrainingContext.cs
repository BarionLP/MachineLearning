﻿using MachineLearning.Data.Entry;
using MachineLearning.Model;
using MachineLearning.Training.Evaluation;
using MachineLearning.Training.Optimization.Layer;
using System.Collections.Immutable;

namespace MachineLearning.Training;

internal sealed class NetworkTrainingContext<TInput, TOutput>(RecordingNetwork<TInput, TOutput> network, TrainingConfig<TInput, TOutput> config)
{
    internal RecordingNetwork<TInput, TOutput> Network = network;
    public TrainingConfig<TInput, TOutput> Config { get; } = config;
    internal ImmutableArray<ILayerOptimizer<Number>> LayerContexts = network.Layers.Select(config.Optimizer.CreateLayerOptimizer).ToImmutableArray();
    internal ILayerOptimizer<Number> OutputLayerContext => LayerContexts[^1];

    public void Train(IEnumerable<DataEntry<TInput, TOutput>> trainingBatch)
    {
        GradientCostReset();
        var dataCounter = 0;

        foreach (var dataPoint in trainingBatch)
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
        Number totalCost = 0;
        var dataCounter = 0;

        foreach (var dataPoint in trainingBatch)
        {
            dataCounter++;
            var result = Update(dataPoint)!;
            if (result.Equals(dataPoint.Expected))
            {
                correctCounter++;
            }
            totalCost += Config.CostFunction.TotalCost(Network.LastOutputWeights, Config.OutputResolver.Expected(dataPoint.Expected));
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
        foreach (var layer in LayerContexts)
        {
            layer.Apply(dataCounter);
        }
    }

    private void GradientCostReset()
    {
        foreach (var layer in LayerContexts)
        {
            layer.GradientCostReset();
        }
    }

    public void FullReset()
    {
        foreach (var layer in LayerContexts)
        {
            layer.FullReset();
        }
    }

    private TOutput Update(DataEntry<TInput, TOutput> data)
    {
        var result = Network.Process(Network.Embedder.Embed(data.Input));

        var nodeValues = OutputLayerContext.CalculateOutputLayerNodeValues(Config.OutputResolver.Expected(data.Expected));
        OutputLayerContext.Update(nodeValues);


        for (int hiddenLayerIndex = LayerContexts.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--)
        {
            var hiddenLayer = LayerContexts[hiddenLayerIndex];
            nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(LayerContexts[hiddenLayerIndex + 1].Layer, nodeValues);
            hiddenLayer.Update(nodeValues);
        }

        return result;
    }
}