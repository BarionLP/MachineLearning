namespace ML.Analyzer.LayerFile.Operations;

internal sealed class OperationFactory(LayerRegistry registry)
{
    private readonly LayerRegistry registry = registry;

    public AddOperation NewAdd(Weights left, Weights right, string resultName)
        => new(left, right, registry.CreateWeights(resultName, left.Dimensions, Location.Snapshot));
    public ActivationOperation NewLeakyReLU(Weights source, string resultName)
        => new(source, registry.CreateWeights(resultName, source.Dimensions, Location.Snapshot));
    public MatrixVectorMultiplyOperation NewMatrixVectorMultiply(Weights matrix, Weights vector, string resultName)
        => new(matrix, vector, registry.CreateWeights(resultName, [matrix.Dimensions[0]], Location.Snapshot));
}