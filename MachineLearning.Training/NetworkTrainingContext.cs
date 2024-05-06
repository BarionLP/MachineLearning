using MachineLearning.Data.Entry;
using MachineLearning.Model;
using MachineLearning.Training.Evaluation;
using MachineLearning.Training.Optimization;
using MachineLearning.Training.Optimization.Layer;
using System.Collections.Immutable;
using System.Diagnostics;

namespace MachineLearning.Training;

internal sealed class NetworkTrainingContext<TInput, TOutput>(RecordingNetwork<TInput, TOutput> network, TrainingConfig<TInput, TOutput> config, IOptimizer<double> optimizer)
{
    internal RecordingNetwork<TInput, TOutput> Network = network;
    public TrainingConfig<TInput, TOutput> Config { get; } = config;
    internal ImmutableArray<ILayerOptimizer<Number>> LayerContexts = network.Layers.Select(optimizer.CreateLayerOptimizer).ToImmutableArray();
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
            totalCost += Config.Optimizer.CostFunction.TotalCost(Network.LastOutputWeights, Config.OutputResolver.Expected(dataPoint.Expected));
        }

        Apply(dataCounter);
        //HasModelErrors();

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

    public void HasModelErrors()
    {
        foreach (var layer in LayerContexts)
        {
            foreach (var ouputNodeIndex in ..layer.Layer.OutputNodeCount)
            {
                if(double.IsNaN(layer.Layer.Biases[ouputNodeIndex]) || double.IsInfinity(layer.Layer.Biases[ouputNodeIndex]))
                {
                    Debug.WriteLine("Model has invalid values!!!");
                }
                foreach (var inputNodeIndex in ..layer.Layer.InputNodeCount){
                    if(double.IsNaN(layer.Layer.Weights[inputNodeIndex, ouputNodeIndex]) || double.IsInfinity(layer.Layer.Weights[inputNodeIndex, ouputNodeIndex]))
                    {
                        Debug.WriteLine("Model has invalid values!!!");
                    }
                }
            }
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