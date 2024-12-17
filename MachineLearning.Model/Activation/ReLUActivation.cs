namespace MachineLearning.Model.Activation;

/// <summary>
/// overall best <br/>
/// use learn rate below 0.1 <br/>
/// usually bad for output layer because values go uncontrollable high <br/>
/// can cause death neurons (better <see cref="LeakyReLUActivation"/>) <br/>
/// </summary>
public sealed class ReLUActivation : ISimdActivationMethod
{
    public static readonly ReLUActivation Instance = new();

    public Weight Activate(Weight input) => float.Max(0, input);
    public SimdVector Activate(SimdVector input) => SimdVectorHelper.Max(SimdVector.Zero, input);

    public Weight Derivative(Weight input) => input > 0 ? 1 : 0;
    public SimdVector Derivative(SimdVector input) => SimdVectorHelper.ConditionalSelect(SimdVectorHelper.GreaterThan(input, SimdVector.Zero), SimdVector.One, SimdVector.Zero);
}
