using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;

namespace MachineLearning.Model;

public interface INetwork<TInput, TData, TOutput, TLayer> where TLayer : ILayer<TData>
{
    public TLayer[] Layers { get; }
    public TLayer OutputLayer { get; }
    public IEmbedder<TInput, TData[], TOutput> Embedder { get; }

    public TOutput Process(TInput input);

    public abstract static INetwork<TInput, TData, TOutput, TLayer> Create(TLayer[] layers, IEmbedder<TInput, TData[], TOutput> embedder); //=> throw new NotImplementedException();
}
