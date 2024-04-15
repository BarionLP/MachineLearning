using Simple.Network.Layer;

namespace Simple.Network;

public interface INetwork<TData, TLayer> where TLayer : ILayer<TData> {
    public TLayer[] Layers { get; }
    public TLayer OutputLayer { get; }
}
