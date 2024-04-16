namespace Simple.Network.Activation;

public interface IActivationMethod {
    public Number Activate(Number input);
    public Number Derivative(Number input);

    public Number[] Activate(Number[] input) {
        var result = new Number[input.Length];
        foreach(int i in ..input.Length) {
            result[i] = Activate(input[i]);
        }
        return result;
    }

    public Number[] Derivative(Number[] input) {
        var result = new Number[input.Length];
        foreach(int i in ..input.Length) {
            result[i] = Derivative(input[i]);
        }
        return result;
    }
}
