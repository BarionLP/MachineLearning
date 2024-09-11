namespace MachineLearning.Model.Activation;

public interface IActivationFunction
{
    public void ActivateTo(Vector input, Vector result);
    public void DerivativeTo(Vector input, Vector result);

    public Vector Activate(Vector input)
    {
        var result = Vector.Create(input.Count);
        ActivateTo(input, result);
        return result;
    }
    public Vector Derivative(Vector input)
    {
        var result = Vector.Create(input.Count);
        DerivativeTo(input, result);
        return result;
    }
}


public interface ISimpleActivationMethod : IActivationFunction
{
    public Weight Activate(Weight input);
    public Weight Derivative(Weight input);

    void IActivationFunction.ActivateTo(Vector input, Vector result) => input.MapTo(Activate, result);
    void IActivationFunction.DerivativeTo(Vector input, Vector result) => input.MapTo(Derivative, result);
}

public interface ISimdActivationMethod : IActivationFunction
{
    public Weight Activate(Weight input);
    public SimdVector Activate(SimdVector input);

    public Weight Derivative(Weight input);
    public SimdVector Derivative(SimdVector input);

    void IActivationFunction.ActivateTo(Vector input, Vector result) => input.MapTo(Activate, Activate, result);
    void IActivationFunction.DerivativeTo(Vector input, Vector result) => input.MapTo(Derivative, Derivative, result);
}