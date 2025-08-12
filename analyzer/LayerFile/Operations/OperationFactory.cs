using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class OperationFactory(LayerRegistry registry)
{
    private readonly LayerRegistry registry = registry;

    public AddOperation NewAdd(Weights left, Weights right, string resultName)
        => new(left, right, registry.CreateWeights(resultName, left.Dimensions, Location.Snapshot));
    public ActivationOperation NewLeakyReLU(Weights source, string resultName)
        => new(source, registry.CreateWeights(resultName, source.Dimensions, Location.Snapshot));
    public Operation NewMultiply(Weights left, Weights right, string resultName)
    {
        return (left.Type, right.Type) switch
        {
            (NumberType.Matrix, NumberType.Vector) when right is RowReferenceWeights rw => new MatrixVectorMultiplyOperation(left, right, registry.CreateResultWeights(resultName, [rw.Matrix.Dimensions[0], left.Dimensions[0]], rw)),
            (NumberType.Matrix, NumberType.Vector) => new MatrixVectorMultiplyOperation(left, right, registry.CreateWeights(resultName, [left.Dimensions[0]], Location.Snapshot)),
            (NumberType.Vector, NumberType.Single) => new VectorSingleMultiplyOperation(left, right, registry.CreateResultWeights(resultName, left)),
            _ => throw new NotImplementedException($"cannot multiply {left} and {right}"),
        };
    }
}

internal sealed class VectorSingleMultiplyOperation(Weights vector, Weights single, Weights result) : Operation
{
    public Weights Vector { get; } = vector.Type is NumberType.Vector ? vector : throw new InvalidOperationException();
    public Weights Single { get; } = single.Type is NumberType.Single ? single : throw new InvalidOperationException();
    public override Weights Result { get; } = result.Dimensions.SequenceEqual(vector.Dimensions) ? result : throw new InvalidOperationException();

    public override void AppendCode(StringBuilder sb)
    {
        sb.AppendLine($$"""{{Vector.PassAccess()}}.MultiplyTo({{Single.PassAccess()}}, {{Result.PassAccess()}})""");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry)
    {
        var resultGradient = registry.GetGradient(Result);
        ops.Add(new VectorSingleMultiplyOperation(resultGradient, Single, registry.CreateWeightsGradient(Vector)));
        ops.Add(new DotProductOperation(resultGradient, Vector, registry.CreateWeightsGradient(Single)));
    }
}
