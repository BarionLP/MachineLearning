namespace MachineLearning.Domain.Numerics.Initialization;

interface IInitializer
{
    public void Initialize(Vector vector);
    public void Initialize(Matrix matrix) => Initialize(matrix.Storage);
}
