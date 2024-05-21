namespace MachineLearning.Domain.Activation;

public interface IActivationMethod
{
    public void Activate(Vector input, Vector result);
    public void Derivative(Vector input, Vector result);

    public Vector Activate(Vector input) {
        var result = Vector.Create(input.Count);
        Activate(input, result);
        return result;
    }
    public Vector Derivative(Vector input) {
        var result = Vector.Create(input.Count);
        Derivative(input, result);
        return result;
    }
}


public interface ISimpleActivationMethod : IActivationMethod
{
    public Weight Activate(Weight input);
    public Weight Derivative(Weight input);

    void IActivationMethod.Activate(Vector input, Vector result) => input.Map(result, Activate);
    void IActivationMethod.Derivative(Vector input, Vector result) => input.Map(result, Derivative);
}