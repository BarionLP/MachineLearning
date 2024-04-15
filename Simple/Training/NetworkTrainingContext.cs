using Simple.Network;
using Simple.Training.Cost;
using System.Collections.Immutable;

namespace Simple.Training;

internal sealed class NetworkTrainingContext(RecordingNetwork network, ICostFunction costFunction) {
    internal RecordingNetwork Network = network;
    internal ImmutableArray<LayerLearningContext> LayerContexts = network.Layers.Select(layer => new LayerLearningContext(layer) { CostFunction = costFunction }).ToImmutableArray();
    internal LayerLearningContext OutputLayerContext => LayerContexts[^1];
    internal ICostFunction CostFunction { get; } = costFunction;

    public void Learn(IEnumerable<DataPoint> trainingBatch, Number learnRate) {
        ClearAllGradients();
        var dataCounter = 0;

        foreach(var dataPoint in trainingBatch) {
            UpdateAllGradients(dataPoint);
            dataCounter++;
        }

        ApplyAllGradients(learnRate / dataCounter); // divide to scale the sum of changes to the average produced by the batch
    }

    private void ApplyAllGradients(Number leanRate) {
        foreach(var layer in LayerContexts) {
            layer.ApplyGradients(leanRate);
        }
    }

    private void ClearAllGradients() {
        foreach(var layer in LayerContexts) {
            layer.ResetGradients();
        }
    }

    private void UpdateAllGradients(DataPoint data) {
        Network.Process(data.Input);

        var nodeValues = OutputLayerContext.CalculateOutputLayerNodeValues(data.Expected);
        OutputLayerContext.UpdateGradients(nodeValues);


        for(int hiddenLayerIndex = LayerContexts.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--) {
            var hiddenLayer = LayerContexts[hiddenLayerIndex];
            nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(LayerContexts[hiddenLayerIndex + 1].Layer, nodeValues);
            hiddenLayer.UpdateGradients(nodeValues);
        }
    }

    public Number Cost(DataPoint data) {
        var output = Network.Process(data.Input);
        return CostFunction.TotalCost(output, data.Expected);
    }

    public Number Cost(DataPoint[] dataPoints) {
        var cost = 0d;
        foreach(var dataPoint in dataPoints) {
            cost += Cost(dataPoint);
        }
        return cost;
    }
}