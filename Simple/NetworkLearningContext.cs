using System.Collections.Immutable;
using System.Numerics;

namespace Simple;

public sealed class NetworkLearningContext(Network network) {
    private readonly Network _network = network;
    private readonly ImmutableArray<LayerLearningContext> _layerContexts = network._layers.Select(layer => new LayerLearningContext(layer)).ToImmutableArray();
    private LayerLearningContext _outputLayerContext => _layerContexts[^1];

    public void Learn(DataPoint<Number[]>[] trainingBatch, Number learnRate, int iterations) {
        foreach(var _ in ..iterations) {
            Learn(trainingBatch, learnRate);
        }
    }
    
    public void LearnBatched(DataPoint<Number[]>[] trainingData, Number learnRate, int batchSize, int iterations) {
        foreach(var _ in ..iterations) {
            LearnBatched(trainingData, learnRate, batchSize);
        }
    }
    
    public void LearnBatched(DataPoint<Number[]>[] trainingData, Number learnRate, int batchSize) {
        Learn(trainingData.GetRandomElements(batchSize), learnRate);
    }
    public void Learn(IEnumerable<DataPoint<Number[]>> trainingBatch, Number learnRate) {
        ClearAllGradients();
        var dataCounter = 0;

        foreach(var dataPoint in trainingBatch){
            UpdateAllGradients(dataPoint);
            dataCounter++;
        }

        ApplyAllGradients(learnRate / dataCounter); // divide to scale the sum of changes to the average produced by the batch
    }

    public Number Test(IEnumerable<DataPoint<Number[]>> testingData) {
        var dataCounter = 0;
        var correctCounter = 0;

        foreach(var dataPoint in testingData) {
            var result = _network.Process(dataPoint.Input);
            if(result[0] > result[1] && dataPoint.Expected[0] > dataPoint.Expected[1]) {
                correctCounter++;
            }
            if(result[0] < result[1] && dataPoint.Expected[0] < dataPoint.Expected[1]) {
                correctCounter++;
            }
            dataCounter++;
        }

        return (Number)correctCounter / dataCounter;
    }

    public void ApplyAllGradients(Number leanRate) {
        foreach(var layer in _layerContexts) {
            layer.ApplyGradients(leanRate);
        }
    }
    
    public void ClearAllGradients() {
        foreach(var layer in _layerContexts) {
            layer.ResetGradients();
        }
    }

    public void UpdateAllGradients(DataPoint<Number[]> data) {
        _network.Process(data.Input);

        var nodeValues = _outputLayerContext.CalculateOutputLayerNodeValues(data.Expected);
        _outputLayerContext.UpdateGradients(nodeValues);


        for(int hiddenLayerIndex = _layerContexts.Length-2; hiddenLayerIndex >= 0; hiddenLayerIndex--) {
            var hiddenLayer = _layerContexts[hiddenLayerIndex];
            nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(_layerContexts[hiddenLayerIndex+1]._layer, nodeValues);
            hiddenLayer.UpdateGradients(nodeValues);
        }
    }

    public Number Cost(DataPoint<Number[]> input) {
        var cost = 0d;

        var output = _network.Process(input.Input);
        foreach(var outputNodeIndex in ..output.Length) {
            cost += NodeHelper.NodeCost(output[outputNodeIndex], input.Expected[outputNodeIndex]);
        }

        return cost;
    }

    public Number Cost(DataPoint<Number[]>[] dataPoints) {
        var cost = 0d;
        foreach(var dataPoint in dataPoints) {
            cost += Cost(dataPoint);
        }
        return cost;
    }
}