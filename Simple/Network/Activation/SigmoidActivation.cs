namespace Simple.Network.Activation;


/// <summary>
/// Produces values between 0 and 1 <br/>
/// Good for Binary Classification <br/>
/// can cause vanishing gradients <br/>
/// </summary>
public sealed class SigmoidActivation : IActivation
{
    public static readonly SigmoidActivation Instance = new();
    public Number Function(Number input) => 1 / (1 + Math.Exp(-input));
    public Number Derivative(Number input)
    {
        var sigmoid = Function(input);
        return sigmoid * (1 - sigmoid);
    }
}
