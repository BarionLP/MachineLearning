using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class PointwiseMultiplyOperation(Weights left, Weights right, Weights result) : Operation
{
    public Weights Left { get; } = left;
    public Weights Right { get; } = right.Dimensions.SequenceEqual(left.Dimensions) ? right : throw new InvalidOperationException($"cannot pointwise multiply {left} and {right}");
    public override Weights Result { get; } = right.Dimensions.SequenceEqual(result.Dimensions) ? result : throw new InvalidOperationException($"{result} cannot store {left} * {right}");

    public override void AppendCode(StringBuilder sb)
    {
        if (ReferenceEquals(Left, Result))
        {
            sb.AppendLine($"{Left.PassAccess()}.PointwiseMultiplyToSelf({Right.PassAccess()});");
        }
        else
        {
            sb.AppendLine($"{Left.PassAccess()}.PointwiseMultiplyTo({Right.PassAccess()}, {Result.PassAccess()});");
        }
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry)
    {
        var rightGradient = registry.CreateWeightsGradient(Right);
        var resultGradient = registry.GetGradient(Result);
        ops.Add(new AddOperation(rightGradient, resultGradient, rightGradient));
        ops.Add(new DefineOperation(resultGradient, registry.CreateWeightsGradient(Left, Location.Pass)));
    }
}
