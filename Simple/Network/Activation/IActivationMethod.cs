namespace Simple.Network.Activation;

public interface IActivationMethod<TData> {
    public TData[] Activate(TData[] input);
    public TData[] Derivative(TData[] input);
}


public interface ISimpleActivationMethod<TData> : IActivationMethod<TData> {
    public TData Activate(TData input);
    public TData Derivative(TData input);

    TData[] IActivationMethod<TData>.Activate(TData[] input)
    {
        var result = new TData[input.Length];
        foreach (int i in ..input.Length)
        {
            result[i] = Activate(input[i]);
        }
        return result;
    }

    TData[] IActivationMethod<TData>.Derivative(TData[] input)
    {
        var result = new TData[input.Length];
        foreach (int i in ..input.Length)
        {
            result[i] = Derivative(input[i]);
        }
        return result;
    }
}