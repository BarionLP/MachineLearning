namespace MachineLearning.Domain.Numerics.Initialization;

public interface IInitializer
{
    public void Initialize(Span<Weight> span);
    public void Initialize(Vector vector) => Initialize(vector.AsSpan());
    public void Initialize(Matrix matrix) => Initialize(matrix.AsSpan());
}
