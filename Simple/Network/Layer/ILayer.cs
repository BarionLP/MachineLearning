using Simple.Network.Activation;

namespace Simple.Network.Layer;

public interface ILayer<TData> {
    public int InputNodeCount { get; }
    public int OutputNodeCount { get; }
    public TData[,] Weights { get; }
    public TData[] Biases { get; }
    public IActivationMethod<TData> ActivationMethod { get; }

    public TData[] Process(TData[] input);

    public virtual static ILayer<TData> Create(TData[,] weights, TData[] biases, IActivationMethod<TData> activationMethod) => throw new NotImplementedException();
}
