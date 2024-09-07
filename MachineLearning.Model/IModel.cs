using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;

namespace MachineLearning.Model;

public interface IModel<TInput, TOutput, TLayer> where TLayer : ILayer
{
    public TLayer[] Layers { get; }
    public TLayer OutputLayer => Layers[^1];
    public IEmbedder<TInput, TOutput> Embedder { get; }

    public TOutput Process(TInput input);
    public Vector Forward(Vector input);

    public abstract static IModel<TInput, TOutput, TLayer> Create(TLayer[] layers, IEmbedder<TInput, TOutput> embedder);
}
