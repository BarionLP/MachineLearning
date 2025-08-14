namespace ML.Analyzer.LayerFile.Operations;

internal sealed class OperationFactory(LayerRegistry registry)
{
    private readonly LayerRegistry registry = registry;

    public Operation NewMultiply(Weights left, Weights right, Weights result, bool? add = null)
    {
        return (left.Type, right.Type) switch
        {
            (NumberType.Matrix, NumberType.Vector) when right is RowReferenceWeights rw => new MatrixVectorMultiplyOperation(left, right, result),
            (NumberType.Matrix, NumberType.Vector) => new MatrixVectorMultiplyOperation(left, right, result),
            (NumberType.Vector, NumberType.Single) => new VectorSingleMultiplyOperation(left, right, result, add),
            _ => throw new NotImplementedException($"cannot multiply {left} and {right}"),
        };
    }
}
