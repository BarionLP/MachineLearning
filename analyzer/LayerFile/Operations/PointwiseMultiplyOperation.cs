using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class PointwiseMultiplyOperation(Weights left, Weights right, Weights result) : Operation
{
    public Weights Left { get; } = left;
    public Weights Right { get; } = right.Dimensions.SequenceEqual(left.Dimensions) ? right : throw new InvalidOperationException($"cannot pointwise multiply {left} and {right}");
    public override Weights Result { get; } = right.Dimensions.SequenceEqual(result.Dimensions) ? result : throw new InvalidOperationException($"{result} cannot store {left} * {right}");

    public override void AppendCode(MethodBodyWriter sb)
    {
        if (ReferenceEquals(Left, Result))
        {
            sb.WriteOperation($"{Left.PassAccess()}.PointwiseMultiply{(Result.Location is Location.Gradients ? "Add" : "")}ToSelf({Right.PassAccess()});");
        }
        else
        {
            sb.WriteOperation($"{Left.PassAccess()}.PointwiseMultiply{(Result.Location is Location.Gradients ? "Add" : "")}To({Right.PassAccess()}, {Result.PassAccess()});");
        }
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry, OperationFactory factory)
    {
        var rightGradient = registry.GetOrCreateGradient(Right);
        var resultGradient = registry.GetGradient(Result);
        ops.Add(new AddOperation(rightGradient, resultGradient, rightGradient));
        ops.Add(new DefineOperation(resultGradient, registry.CreateWeightsGradient(Left, Location.Pass)));
    }
}
