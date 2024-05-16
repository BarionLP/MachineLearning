namespace MachineLearning.Domain.Activation;

/// <summary>
/// outputs values between -1 and 1 (adjust output resolver!) <br/>
/// can cause vanishing gradients (often better than <see cref="SigmoidActivation"/> because centered around zero)
/// </summary>
public sealed class TanhActivation : ISimpleActivationMethod<double>
{
    public static readonly TanhActivation Instance = new();
    public double Activate(double input) => Math.Tanh(input);
    public double Derivative(double input) => 1 - Math.Pow(Math.Tanh(input), 2);
}
