namespace Simple.Network.Activation;
/// <summary>
/// use learn rate below 0.1 <br/>
/// usually bad for output layer because values can go infinite high <br/>
/// helps with death neurons in <see cref="ReLUActivation"/> <br/>
/// </summary>
/// <param name="alpha">slope for x &lt; 0</param>
public sealed class LeakyReLUActivation(Number alpha = 0.01) : IActivationMethod {
    public static readonly LeakyReLUActivation Instance = new();
    private readonly Number alpha = alpha;
    public Number Activate(Number input) => input > 0 ? input : alpha * input;
    public Number Derivative(Number input) => input > 0 ? 1 : alpha;
}
