using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;
using MachineLearning.Model.Layer.Snapshot;
using System.Collections.Immutable;

namespace MachineLearning.Model;

public sealed class EmbeddedModel<TInput, TOutput>(SimpleModel model, IEmbedder<TInput, TOutput> embedder) : IGenericModel<TInput, TOutput>
{
    public SimpleModel InnerModel { get; } = model;
    public IEmbedder<TInput, TOutput> Embedder { get; } = embedder;
    public IEmbeddingLayer<TInput> InputLayer => Embedder;
    public IUnembeddingLayer<TOutput> OutputLayer => Embedder;
    public IEnumerable<ILayer> Layers { get;  } = [embedder,..model.Layers, embedder];
    public uint ParameterCount => InnerModel.ParameterCount;

    ISimpleModel<SimpleLayer> IGenericModel<TInput, TOutput>.InnerModel => InnerModel;

    public (TOutput output, Weight confidence) Process(TInput input) => Embedder.Unembed(InnerModel.Forward(Embedder.Embed(input)));
    public (TOutput output, double confidence) Forward(TInput input) => Process(input);
    public (TOutput output, int outIndex, Vector weights) Forward(TInput input, ImmutableArray<ILayerSnapshot> snapshots){
        var weights = InnerModel.Forward(Embedder.Embed(input));
        var (result, confidence) = Embedder.Unembed(weights);
        return (result, weights.AsSpan().IndexOf(confidence), weights);
    }

    public override string ToString() => $"Embedded {InnerModel}";
}

public sealed class SimpleModel(ImmutableArray<SimpleLayer> layers) : ISimpleModel<SimpleLayer>
{
    public ImmutableArray<SimpleLayer> Layers { get; } = layers;
    public SimpleLayer OutputLayer => Layers[^1];
    public uint ParameterCount => (uint)Layers.Sum(l => l.ParameterCount);


    public Vector Forward(Vector input)
    {
        foreach (var layer in Layers)
        {
            input = layer.Forward(input);
        }
        return input;
    }

    public Vector Forward(Vector input, IEnumerable<ILayerSnapshot> snapshots)
    {
        foreach (var (layer, snapshot) in Layers.Zip(snapshots))
        {
            input = layer.Forward(input, snapshot);
        }
        return input;
    }

    public override string ToString() => $"Simple Feed Forward Model ({Layers.Length} Layers, {ParameterCount} Weights)";
}
