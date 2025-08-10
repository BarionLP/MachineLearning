using System.Collections.Generic;
using System.Security.Cryptography;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class MatrixVectorMultiplyOperation(Weights matrix, Weights vector, Weights result) : Operation
{
    public Weights Matrix { get; } = matrix.Type is NumberType.Matrix ? matrix : throw new InvalidOperationException($"{matrix} is not a matrix");
    public Weights Vector { get; } = vector.Type is NumberType.Vector ? vector : throw new InvalidOperationException($"{vector} is not a vector");
    public override Weights Result { get; } = result;

    public override void AppendCode(StringBuilder sb)
    {
        sb.AppendLine($"{Matrix.PassAccess()}.MultiplyTo({Vector.PassAccess()}, {Result.PassAccess()});");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry)
    {
        ops.Add(new VectorVectorMultiplyMatrixOperation(registry.GetGradient(Result), Vector, registry.CreateWeightsGradient(Matrix)));
        ops.Add(new MatrixTransposedVectorMultiplyOperation(Matrix, registry.GetGradient(Result), registry.CreateWeightsGradient(Vector)));
    }
}

internal sealed class MatrixTransposedVectorMultiplyOperation(Weights matrix, Weights vector, Weights result) : Operation
{
    public Weights Matrix { get; } = matrix.Type is NumberType.Matrix ? matrix : throw new InvalidOperationException($"{matrix} is not a matrix");
    public Weights Vector { get; } = vector.Type is NumberType.Vector ? vector : throw new InvalidOperationException($"{vector} is not a vector");
    public override Weights Result { get; } = result;

    public override void AppendCode(StringBuilder sb)
    {
        sb.AppendLine($"{Matrix.PassAccess()}.MultiplyTransposedTo({Vector.PassAccess()}, {Result.PassAccess()});");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry)
    {
        throw new NotImplementedException();        
    }
}
