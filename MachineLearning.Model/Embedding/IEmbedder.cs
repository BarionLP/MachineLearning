using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;

namespace MachineLearning.Model.Embedding;

public interface IEmbedder<in TInput, TOutput> : IEmbeddingLayer<TInput>, IUnembeddingLayer<TOutput> where TInput : allows ref struct
{
    public Vector Embed(TInput input);
    public Vector Embed(TInput input, ILayerSnapshot snapshot);
    public (TOutput output, Weight confidence) Unembed(Vector input);
    public (TOutput output, int index, Vector weights) Unembed(Vector input, ILayerSnapshot snapshot);

    int IEmbeddingLayer<TInput>.OutputNodeCount => 0;
    int IUnembeddingLayer<TOutput>.InputNodeCount => 0;
    uint ILayer.ParameterCount => 0;

    Vector IEmbeddingLayer<TInput>.Forward(TInput input) => Embed(input);
    Vector IEmbeddingLayer<TInput>.Forward(TInput input, ILayerSnapshot snapshot) => Embed(input, snapshot);

    (TOutput output, Weight confidence) IUnembeddingLayer<TOutput>.Forward(Vector input) => Unembed(input);
    (TOutput output, int index, Vector weights) IUnembeddingLayer<TOutput>.Forward(Vector input, ILayerSnapshot snapshot) => Unembed(input, snapshot);
}
