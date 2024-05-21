namespace MachineLearning.Domain.Activation;

/// <summary>
/// overall best <br/>
/// use learn rate below 0.1 <br/>
/// usually bad for output layer because values go uncontrollable high <br/>
/// can cause death neurons (better <see cref="LeakyReLUActivation"/>) <br/>
/// </summary>
public sealed class ReLUActivation : ISimpleActivationMethod
{
    public static readonly ReLUActivation Instance = new();
    public Weight Activate(Weight input) => Math.Max(0, input);
    public Weight Derivative(Weight input) => input > 0 ? 1 : 0;
}
