using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Model.Layer;

public interface IEmbeddingLayer<in TInput> : ILayer where TInput : allows ref struct
{
    public int OutputNodeCount { get; }

    public Vector Process(TInput input);
    public Vector Process(TInput input, ILayerSnapshot snapshot);
}

public interface IUnembeddingLayer<TOutput> : ILayer
{
    public int InputNodeCount { get; }

    public (TOutput output, Weight confidence) Process(Vector input);
    public (TOutput output, Weight confidence, Vector weights) Process(Vector input, ILayerSnapshot snapshot);
}

public interface ILayer
{
    public long ParameterCount { get; }

    public ILayerSnapshot CreateSnapshot();
};