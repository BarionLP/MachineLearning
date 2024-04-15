using Simple.Network.Activation;
using Simple.Network.Layer;

namespace Simple.Network;

public sealed class RecordingNetwork : INetwork<Number, RecordingLayer>{
    public RecordingLayer[] Layers { get;  }
    public RecordingLayer OutputLayer => Layers[^1];
    public IActivation ActivationMethod { get; init; } = SigmoidActivation.Instance;


    public RecordingNetwork(params int[] layerSizes){
        Layers = new RecordingLayer[layerSizes.Length - 1]; // no need to define the input layer
        foreach (var i in ..Layers.Length){
            Layers[i] = new RecordingLayer(layerSizes[i], layerSizes[i + 1]) { ActivationMethod = ActivationMethod };
        }
    }

    public Number[] Process(Number[] input){
        foreach (var layer in Layers){
            input = layer.Process(input);
        }
        return input;
    }
}