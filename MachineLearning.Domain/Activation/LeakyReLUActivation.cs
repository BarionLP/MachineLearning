namespace MachineLearning.Domain.Activation;
/// <summary>
/// use learn rate below 0.1 <br/>
/// usually bad for output layer because values can go infinite high <br/>
/// helps with death neurons in <see cref="ReLUActivation"/> <br/>
/// </summary>
/// <param name="alpha">slope for x &lt; 0</param>
public sealed class LeakyReLUActivation(Weight alpha = 0.01) : ISimdActivationMethod
{
    public static readonly LeakyReLUActivation Instance = new();
    private readonly Weight alpha = alpha;
    public Weight Activate(Weight input) => input > 0 ? input : alpha * input;
    public SimdVector Activate(SimdVector input) => SimdVectorHelper.ConditionalSelect(SimdVectorHelper.GreaterThan(input, SimdVector.Zero), input, input * alpha);

    public Weight Derivative(Weight input) => input > 0 ? 1 : alpha;
    public SimdVector Derivative(SimdVector input) => SimdVectorHelper.ConditionalSelect(SimdVectorHelper.GreaterThan(input, SimdVector.Zero), SimdVector.One, new SimdVector(alpha));
}
