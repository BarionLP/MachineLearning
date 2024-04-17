using Simple.Network;
using Simple.Training.Cost;
using System.Collections.Immutable;

namespace Simple.Training;

internal sealed class NetworkTrainingContext<TInput, TOutput>(RecordingNetwork<TInput, TOutput> network, ICostFunction costFunction, IInputDataNoise<Number> inputNoise, IOutputResolver<TOutput, Number[]> outputResolver) {
    internal RecordingNetwork<TInput, TOutput> Network = network;
    internal ImmutableArray<LayerLearningContext> LayerContexts = network.Layers.Select(layer => new LayerLearningContext(layer) { CostFunction = costFunction }).ToImmutableArray();
    internal LayerLearningContext OutputLayerContext => LayerContexts[^1];
    internal ICostFunction CostFunction { get; } = costFunction;
    public IInputDataNoise<Number> InputNoise { get; } = inputNoise;
    internal IOutputResolver<TOutput, Number[]> OutputResolver { get; } = outputResolver;

    public void Learn(IEnumerable<DataPoint<TInput, TOutput>> trainingBatch, Number learnRate) {
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

    private void UpdateAllGradients(DataPoint<TInput, TOutput> data) {
        Network.Process(InputNoise.Apply(Network.Embedder.Embed(data.Input)));

        var nodeValues = OutputLayerContext.CalculateOutputLayerNodeValues(OutputResolver.Expected(data.Expected));
        OutputLayerContext.UpdateGradients(nodeValues);


        for(int hiddenLayerIndex = LayerContexts.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--) {
            var hiddenLayer = LayerContexts[hiddenLayerIndex];
            nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(LayerContexts[hiddenLayerIndex + 1].Layer, nodeValues);
            hiddenLayer.UpdateGradients(nodeValues);
        }
    }
}