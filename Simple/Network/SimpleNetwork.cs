using Simple.Network.Layer;

namespace Simple.Network;

public sealed class SimpleNetwork(ILayer<Number>[] layers) : INetwork<Number, ILayer<Number>>{
    public ILayer<Number>[] Layers { get; } = layers;
    public ILayer<Number> OutputLayer => Layers[^1];

    public Number[] Process(Number[] input){
        foreach (var layer in Layers){
            input = layer.Process(input);
        }
        return input;
    }
}
