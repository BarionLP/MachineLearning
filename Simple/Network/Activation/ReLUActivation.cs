namespace Simple.Network.Activation;

/// <summary>
/// overall best <br/>
/// can cause death neurons (better <see cref="LeakyReLUActivation"/>)
/// </summary>
public sealed class ReLUActivation : IActivation
{
    public static readonly ReLUActivation Instance = new();
    public Number Function(Number input) => Math.Max(0, input);
    public Number Derivative(Number input) => input > 0 ? 1 : 0;
}
