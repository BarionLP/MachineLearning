using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;
using System.Collections.Immutable;

namespace MachineLearning.Model;


public sealed class EmbeddedModel<TInput, TOutput>(SimpleModel model, IEmbedder<TInput, TOutput> embedder)
{
    public SimpleModel InternalModel { get; } = model;
    public IEmbedder<TInput, TOutput> Embedder { get; } = embedder;

    public TOutput Process(TInput input) => Embedder.UnEmbed(InternalModel.Forward(Embedder.Embed(input)));
}

public sealed class SimpleModel(ImmutableArray<SimpleLayer> layers)
{
    public ImmutableArray<SimpleLayer> Layers { get; } = layers;

    public Vector Forward(Vector weights)
    {
        foreach(var layer in Layers)
        {
            weights = layer.Forward(weights);
        }
        return weights;
    }
}
