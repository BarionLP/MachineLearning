namespace Simple.Network.Activation;

/// <summary>
/// outputs values between -1 and 1 <br/>
/// can cause vanishing gradients (often better than <see cref="SigmoidActivation"/> because centered around zero)
/// </summary>
public sealed class TanhActivation : IActivation
{
    public static readonly TanhActivation Instance = new();
    public Number Function(Number input) => Math.Tanh(input);
    public Number Derivative(Number input) => 1 - Math.Pow(Math.Tanh(input), 2);
}
