using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class VectorVectorMultiplyMatrixOperation(Weights rowVector, Weights columnVector, Weights result) : Operation
{
    public Weights RowVector { get; } = rowVector.Type is NumberType.Vector ? rowVector : throw new InvalidOperationException($"{rowVector} is not a vector");
    public Weights ColumnVector { get; } = columnVector.Type is NumberType.Vector ? columnVector : throw new InvalidOperationException($"{columnVector} is not a vector");
    public override Weights Result { get; } = result;

    public override void AppendCode(MethodBodyWriter sb)
    {
        sb.WriteOperation($"VectorHelper.MultiplyToMatrix{(Result.Location is Location.Gradients ? "Add" : "")}To({RowVector.PassAccess()}, {ColumnVector.PassAccess()}, {Result.PassAccess()});");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry, OperationFactory factory)
    {
        throw new NotImplementedException();
    }
}
