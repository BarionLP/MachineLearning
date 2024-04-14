namespace Simple.Activation; 

public interface IActivation {
    public Number[] Function(Number[] input);

    public Number Function(Number input);
    public Number Derivative(Number input);
}

public sealed class SigmoidActivation : IActivation{
    public static readonly SigmoidActivation Instance = new();

    public Number[] Function(Number[] input) {
        foreach(int i in ..input.Length) {
            input[i] = Function(input[i]);
        }
        return input;
    }

    public Number Function(Number input) => 1 / (1 + Math.Exp(-input));
    public Number Derivative(Number input) {
        var value = Function(input);
        return value * (1 - value);
    }
}