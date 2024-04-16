using Simple.Network.Embedding;
using Simple.Network.Layer;

namespace Simple.Network;

public interface INetwork<in TInput, TData, out TOutput, TLayer> where TLayer : ILayer<TData> {
    public TLayer[] Layers { get; }
    public TLayer OutputLayer { get; }
    public IEmbedder<TInput, TData[], TOutput> Embedder { get; }

    public TOutput Process(TInput input);
}
