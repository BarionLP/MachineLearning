namespace MachineLearning.Model.Activation;

public interface IActivationFunction
{
    public void Activate(Vector input, Vector result);
    public void Derivative(Vector input, Vector result);

    public Vector Activate(Vector input)
    {
        var result = Vector.Create(input.Count);
        Activate(input, result);
        return result;
    }
    public Vector Derivative(Vector input)
    {
        var result = Vector.Create(input.Count);
        Derivative(input, result);
        return result;
    }
}


public interface ISimpleActivationMethod : IActivationFunction
{
    public Weight Activate(Weight input);
    public Weight Derivative(Weight input);

    void IActivationFunction.Activate(Vector input, Vector result) => input.Map(Activate, result);
    void IActivationFunction.Derivative(Vector input, Vector result) => input.Map(Derivative, result);
}

public interface ISimdActivationMethod : IActivationFunction
{
    public Weight Activate(Weight input);
    public SimdVector Activate(SimdVector input);

    public Weight Derivative(Weight input);
    public SimdVector Derivative(SimdVector input);

    void IActivationFunction.Activate(Vector input, Vector result) => input.Map(Activate, Activate, result);
    void IActivationFunction.Derivative(Vector input, Vector result) => input.Map(Derivative, Derivative, result);
}