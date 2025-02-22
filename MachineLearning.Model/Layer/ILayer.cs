using System.Formats.Tar;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Model.Layer;

public interface IEmbeddingLayer<TInput> : ILayer<TInput, Vector, ILayerSnapshot>
{
    public int OutputNodeCount { get; }

    public Vector Process(TInput input);
    public Vector Process(TInput input, ILayerSnapshot snapshot);
}

public interface IUnembeddingLayer<TOutput> : ILayer<Vector, TOutput, ILayerSnapshot>
{
    public int InputNodeCount { get; }

    public (TOutput output, Weight confidence) Process(Vector input);
    public (TOutput output, Weight confidence, Vector weights) Process(Vector input, ILayerSnapshot snapshot);
}

public interface ILayer
{
    public long WeightCount { get; }

    public ILayerSnapshot CreateSnapshot();
};

public interface ILayer<TIn, TOut, TSnapshot> : ILayer where TSnapshot : ILayerSnapshot;
public interface ILayer<TArch, TSnapshot> : ILayer<TArch, TArch, TSnapshot> where TSnapshot : ILayerSnapshot
{
    public TArch Forward(TArch input, TSnapshot snapshot);
}