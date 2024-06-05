using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;

namespace MachineLearning.Model;

public interface INetwork<TInput, TOutput, TLayer> where TLayer : ILayer
{
    public TLayer[] Layers { get; }
    public TLayer OutputLayer { get; }
    public IEmbedder<TInput, TOutput> Embedder { get; }

    public TOutput Process(TInput input);
    public Vector Forward(Vector input);

    public abstract static INetwork<TInput, TOutput, TLayer> Create(TLayer[] layers, IEmbedder<TInput, TOutput> embedder); //=> throw new NotImplementedException();
}
