namespace MachineLearning.Model.Layer.Snapshot;

public interface IGradients
{
    public static IGradients Empty { get; } = new EmptyGradients();
    public void Add(IGradients other);

    public void Reset();
}

file sealed record EmptyGradients : IGradients
{
    public void Add(IGradients other)
    {
        Guard.Is<EmptyGradients>(other);
    }

    public void Reset()
    {
    }
}