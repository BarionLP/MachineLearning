using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class MatrixVectorMultiplyOperation(Weights matrix, Weights vector, Weights result) : Operation
{
    public Weights Matrix { get; } = matrix.Type is NumberType.Matrix ? matrix : throw new InvalidOperationException($"{matrix} is not a matrix");
    public Weights Vector { get; } = vector.Type is NumberType.Vector ? vector : throw new InvalidOperationException($"{vector} is not a vector");
    public override Weights Result { get; } = result;

    public override void AppendCode(MethodBodyWriter sb)
    {
        sb.WriteOperation($"{Matrix.PassAccess()}.Multiply{(Result.Location is Location.Gradients ? "Add" : "")}To({Vector.PassAccess()}, {Result.PassAccess()});");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry, OperationFactory factory)
    {
        ops.Add(new VectorVectorMultiplyMatrixOperation(registry.GetGradient(Result), Vector, registry.GetOrCreateGradient(Matrix)));
        ops.Add(new MatrixTransposedVectorMultiplyOperation(Matrix, registry.GetGradient(Result), registry.GetOrCreateGradient(Vector)));
    }
}

internal sealed class MatrixTransposedVectorMultiplyOperation(Weights matrix, Weights vector, Weights result) : Operation
{
    public Weights Matrix { get; } = matrix.Type is NumberType.Matrix ? matrix : throw new InvalidOperationException($"{matrix} is not a matrix");
    public Weights Vector { get; } = vector.Type is NumberType.Vector ? vector : throw new InvalidOperationException($"{vector} is not a vector");
    public override Weights Result { get; } = result;

    public override void AppendCode(MethodBodyWriter sb)
    {
        sb.WriteOperation($"{Matrix.PassAccess()}.MultiplyTransposed{(Result.Location is Location.Gradients ? "Add" : "")}To({Vector.PassAccess()}, {Result.PassAccess()});");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry, OperationFactory factory)
    {
        throw new NotImplementedException();        
    }
}
