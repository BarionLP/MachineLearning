using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Model.Layer;

public interface IEmbeddingLayer<in TInput> : ILayer where TInput : allows ref struct
{
    public int OutputNodeCount { get; }

    public Vector Forward(TInput input);
    public Vector Forward(TInput input, ILayerSnapshot snapshot);
}

public interface IUnembeddingLayer<TOutput> : ILayer
{
    public int InputNodeCount { get; }

    public (TOutput output, Weight confidence) Forward(Vector input);
    public (TOutput output, int index, Vector weights) Forward(Vector input, ILayerSnapshot snapshot);
}

public interface ILayer
{
    public long ParameterCount { get; }
};