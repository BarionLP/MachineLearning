namespace Simple.Network.Activation;

public interface IActivationMethod<TData> {
    public TData Activate(TData input);
    public TData Derivative(TData input);

    public TData[] Activate(TData[] input) {
        var result = new TData[input.Length];
        foreach(int i in ..input.Length) {
            result[i] = Activate(input[i]);
        }
        return result;
    }

    public TData[] Derivative(TData[] input) {
        var result = new TData[input.Length];
        foreach(int i in ..input.Length) {
            result[i] = Derivative(input[i]);
        }
        return result;
    }
}
