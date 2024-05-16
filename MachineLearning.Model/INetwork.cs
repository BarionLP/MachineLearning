using MachineLearning.Model.Embedding;
using MachineLearning.Model.Layer;

namespace MachineLearning.Model;

public interface INetwork<TInput, TWeight, TOutput, TLayer> where TLayer : ILayer<TWeight> where TWeight : struct, IEquatable<TWeight>, IFormattable
{
    public TLayer[] Layers { get; }
    public TLayer OutputLayer { get; }
    public IEmbedder<TInput, Vector<TWeight>, TOutput> Embedder { get; }

    public TOutput Process(TInput input);
    public Vector<TWeight> Forward(Vector<TWeight> input);

    public abstract static INetwork<TInput, TWeight, TOutput, TLayer> Create(TLayer[] layers, IEmbedder<TInput, Vector<TWeight>, TOutput> embedder); //=> throw new NotImplementedException();
}
