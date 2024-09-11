using System.Collections.Immutable;
using MachineLearning.Model.Layer;

namespace MachineLearning.Model;

public interface ISimpleModel<TLayer> : IModel where TLayer : ILayer
{
    public ImmutableArray<TLayer> Layers { get; }
    public TLayer OutputLayer { get; }

    public Vector Forward(Vector input);
}

public interface IModel
{
    public uint ParameterCount { get; }
}

public interface IEmbeddedModel<in TInput, TOutput> : IModel
{
    public IEmbeddingLayer<TInput> InputLayer { get; }
    public ISimpleModel<SimpleLayer> InnerModel { get; }
    public IUnembeddingLayer<TOutput> OutputLayer { get; }
    public (TOutput output, Weight confidence) Forward(TInput input);
}

public sealed class FeedForwardModel<TInput, TOutput> : IEmbeddedModel<TInput, TOutput>
{
    public required IEmbeddingLayer<TInput> InputLayer { get; init; }
    public required ISimpleModel<SimpleLayer> InnerModel { get; init; }
    public required IUnembeddingLayer<TOutput> OutputLayer { get; init; }
    public uint ParameterCount => InputLayer.ParameterCount + InnerModel.ParameterCount + OutputLayer.ParameterCount;

    public (TOutput output, Weight confidence) Forward(TInput input)
    {
        var vector = InputLayer.Forward(input);
        vector = InnerModel.Forward(vector);
        return OutputLayer.Forward(vector);
    }
}