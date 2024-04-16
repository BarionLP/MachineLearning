namespace Simple.Network.Activation;

/// <summary>
/// outputs values between -1 and 1 <br/>
/// can cause vanishing gradients (often better than <see cref="SigmoidActivation"/> because centered around zero)
/// </summary>
public sealed class TanhActivation : IActivationMethod
{
    public static readonly TanhActivation Instance = new();
    public Number Activate(Number input) => Math.Tanh(input);
    public Number Derivative(Number input) => 1 - Math.Pow(Math.Tanh(input), 2);
}
