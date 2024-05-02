using Simple.Network;
using Simple.Training.Optimization;
using Simple.Training.Optimization.Layer;
using System.Collections.Immutable;

namespace Simple.Training;

internal sealed class NetworkTrainingContext<TInput, TOutput>(RecordingNetwork<TInput, TOutput> network, IOptimizerConfig<Number> optimizer, IOutputResolver<TOutput, Number[]> outputResolver) {
    internal RecordingNetwork<TInput, TOutput> Network = network;
    internal ImmutableArray<ILayerOptimizer<Number>> LayerContexts = network.Layers.Select(optimizer.CreateLayerOptimizer).ToImmutableArray();
    internal ILayerOptimizer<Number> OutputLayerContext => LayerContexts[^1];
    internal IOptimizerConfig<Number> Optimizer { get; } = optimizer;
    internal IOutputResolver<TOutput, Number[]> OutputResolver { get; } = outputResolver;

    public void Learn(IEnumerable<DataPoint<TInput, TOutput>> trainingBatch) {
        ClearAllGradients();
        var dataCounter = 0;

        foreach(var dataPoint in trainingBatch) {
            UpdateAllGradients(dataPoint);
            dataCounter++;
        }

        ApplyAllGradients(dataCounter);
    }

    private void ApplyAllGradients(int dataCounter) {
        foreach(var layer in LayerContexts) {
            layer.Apply(Optimizer, dataCounter);
        }
    }

    private void ClearAllGradients() {
        foreach(var layer in LayerContexts) {
            layer.Reset();
        }
    }

    private void UpdateAllGradients(DataPoint<TInput, TOutput> data) {
        Network.Process(Network.Embedder.Embed(data.Input));

        var nodeValues = OutputLayerContext.CalculateOutputLayerNodeValues(OutputResolver.Expected(data.Expected));
        OutputLayerContext.Update(nodeValues);


        for(int hiddenLayerIndex = LayerContexts.Length - 2; hiddenLayerIndex >= 0; hiddenLayerIndex--) {
            var hiddenLayer = LayerContexts[hiddenLayerIndex];
            nodeValues = hiddenLayer.CalculateHiddenLayerNodeValues(LayerContexts[hiddenLayerIndex + 1].Layer, nodeValues);
            hiddenLayer.Update(nodeValues);
        }
    }
}