using System.Collections.Immutable;
using System.Diagnostics;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Model;

public interface IModel
{
    public uint ParameterCount { get; }
}

public interface IEmbeddedModel<in TInput, TOutput> : IModel
{
    public IEmbeddingLayer<TInput> InputLayer { get; }
    public ImmutableArray<SimpleLayer> HiddenLayers { get; }
    public IUnembeddingLayer<TOutput> OutputLayer { get; }
    public IEnumerable<ILayer> Layers { get; }
    public (TOutput output, Weight confidence) Forward(TInput input);
    public (TOutput output, int outIndex, Vector weights) Forward(TInput input, ImmutableArray<ILayerSnapshot> snapshots);
}

public sealed class FeedForwardModel<TInput, TOutput> : IEmbeddedModel<TInput, TOutput>
{
    public required IEmbeddingLayer<TInput> InputLayer { get; init; }
    public required ImmutableArray<SimpleLayer> HiddenLayers { get; init; }
    public required IUnembeddingLayer<TOutput> OutputLayer { get; init; }
    public IEnumerable<ILayer> Layers => [InputLayer, .. HiddenLayers, OutputLayer];
    public int LayerCount => HiddenLayers.Length + 2;
    public uint ParameterCount => InputLayer.ParameterCount + (uint)HiddenLayers.Sum(l => l.ParameterCount) + OutputLayer.ParameterCount;

    public (TOutput output, Weight confidence) Forward(TInput input)
    {
        var vector = InputLayer.Forward(input);
        foreach(var layer in HiddenLayers)
        {
            vector = layer.Forward(vector);
        }
        return OutputLayer.Forward(vector);
    }
    public (TOutput output, int outIndex, Vector weights) Forward(TInput input, ImmutableArray<ILayerSnapshot> snapshots) {
        Debug.Assert(snapshots.Length == LayerCount);
        var vector = InputLayer.Forward(input, snapshots[0]);
        foreach(var (layer, snapshot) in HiddenLayers.Zip(snapshots.Skip(1)))
        {
            vector = layer.Forward(vector, snapshot);
        }
        return OutputLayer.Forward(vector, snapshots[^1]);
    }

    public override string ToString() 
        => $"Generic Feed Forward Model ({LayerCount} Layers, {ParameterCount} Weights)";
}