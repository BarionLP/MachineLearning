using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;
using System.Collections.Immutable;

namespace MachineLearning.Model;

public sealed class EmbeddedModel<TInput, TOutput>(SimpleModel model, IEmbedder<TInput, TOutput> embedder)
{
    public SimpleModel InternalModel { get; } = model;
    public IEmbedder<TInput, TOutput> Embedder { get; } = embedder;

    public (TOutput output, Weight confidence) Process(TInput input) => Embedder.Unembed(InternalModel.Forward(Embedder.Embed(input)));

    public override string ToString() => $"Embedded {InternalModel}";
}

public sealed class SimpleModel(ImmutableArray<SimpleLayer> layers) : ISimpleModel<SimpleLayer>
{
    public ImmutableArray<SimpleLayer> Layers { get; } = layers;
    public SimpleLayer OutputLayer => Layers[^1];
    public uint ParameterCount => (uint)Layers.Sum(l => l.ParameterCount);


    public Vector Forward(Vector weights)
    {
        foreach (var layer in Layers)
        {
            weights = layer.Forward(weights);
        }
        return weights;
    }

    public override string ToString() => $"Simple Feed Forward Model ({Layers.Length} Layers, {ParameterCount} Weights)";
}
