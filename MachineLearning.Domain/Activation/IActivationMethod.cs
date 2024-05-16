namespace MachineLearning.Domain.Activation;

public interface IActivationMethod<TData> where TData : struct, IEquatable<TData>, IFormattable
{
    public Vector<TData> Activate(Vector<TData> input);
    public Vector<TData> Derivative(Vector<TData> input);
}


public interface ISimpleActivationMethod<TData> : IActivationMethod<TData> where TData : struct, IEquatable<TData>, IFormattable
{
    public TData Activate(TData input);
    public TData Derivative(TData input);

    Vector<TData> IActivationMethod<TData>.Activate(Vector<TData> input) => input.Map(Activate);
    Vector<TData> IActivationMethod<TData>.Derivative(Vector<TData> input) => input.Map(Derivative);
}