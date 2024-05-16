namespace MachineLearning.Domain.Activation;
/// <summary>
/// use learn rate below 0.1 <br/>
/// usually bad for output layer because values can go infinite high <br/>
/// helps with death neurons in <see cref="ReLUActivation"/> <br/>
/// </summary>
/// <param name="alpha">slope for x &lt; 0</param>
public sealed class LeakyReLUActivation(double alpha = 0.01) : ISimpleActivationMethod<double>
{
    public static readonly LeakyReLUActivation Instance = new();
    private readonly double alpha = alpha;
    public double Activate(double input) => input > 0 ? input : alpha * input;
    public double Derivative(double input) => input > 0 ? 1 : alpha;
}
