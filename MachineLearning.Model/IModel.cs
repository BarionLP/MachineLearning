using System.Collections.Immutable;
using System.Diagnostics;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Model;

public interface ISimpleModel<TLayer> : IModel where TLayer : ILayer
{
    public ImmutableArray<TLayer> Layers { get; }
    public TLayer OutputLayer { get; }

    public Vector Forward(Vector input);
    public Vector Forward(Vector input, IEnumerable<ILayerSnapshot> snapshots);

}

public interface IModel
{
    public uint ParameterCount { get; }
}

public interface IGenericModel<in TInput, TOutput> : IModel
{
    public IEmbeddingLayer<TInput> InputLayer { get; }
    public ISimpleModel<SimpleLayer> InnerModel { get; }
    public IUnembeddingLayer<TOutput> OutputLayer { get; }
    public IEnumerable<ILayer> Layers { get; }
    public (TOutput output, Weight confidence) Forward(TInput input);
    public (TOutput output, int outIndex, Vector weights) Forward(TInput input, ImmutableArray<ILayerSnapshot> snapshots);
}

public sealed class FeedForwardModel<TInput, TOutput> : IGenericModel<TInput, TOutput>
{
    public required IEmbeddingLayer<TInput> InputLayer { get; init; }
    public required ISimpleModel<SimpleLayer> InnerModel { get; init; }
    public required IUnembeddingLayer<TOutput> OutputLayer { get; init; }
    public IEnumerable<ILayer> Layers => [InputLayer, .. InnerModel.Layers, OutputLayer];
    public uint ParameterCount => InputLayer.ParameterCount + InnerModel.ParameterCount + OutputLayer.ParameterCount;

    public (TOutput output, Weight confidence) Forward(TInput input)
    {
        var vector = InputLayer.Forward(input);
        vector = InnerModel.Forward(vector);
        return OutputLayer.Forward(vector);
    }
    public (TOutput output, int outIndex, Vector weights) Forward(TInput input, ImmutableArray<ILayerSnapshot> snapshots) {
        Debug.Assert(snapshots.Length == InnerModel.Layers.Length + 2);
        var vector = InputLayer.Forward(input, snapshots[0]);
        vector = InnerModel.Forward(vector, snapshots.Skip(1).Take(InnerModel.Layers.Length));
        return OutputLayer.Forward(vector, snapshots[^1]);
    }

    public override string ToString() 
        => $"Generic Feed Forward Model ({InnerModel.Layers.Length + 2} Layers, {ParameterCount} Weights)";
}