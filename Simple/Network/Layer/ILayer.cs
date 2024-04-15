using Simple.Network.Activation;

namespace Simple.Network.Layer;

public interface ILayer<TData> {
    public int InputNodeCount { get; }
    public int OutputNodeCount { get; }
    public TData[,] Weights { get; }
    public TData[] Biases { get; }
    public IActivation ActivationMethod { get; }

    public TData[] Process(TData[] input);
}
