namespace Simple.Network.Activation;

/// <summary>
/// overall best <br/>
/// use learn rate below 0.1
/// can cause death neurons (better <see cref="LeakyReLUActivation"/>)
/// </summary>
public sealed class ReLUActivation : IActivationMethod{
    public static readonly ReLUActivation Instance = new();
    public Number Activate(Number input) => Math.Max(0, input);
    public Number Derivative(Number input) => input > 0 ? 1 : 0;
}
