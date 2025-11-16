using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class ConstructAttentionMatrixOperation(Weights queries, Weights keys, Weights result) : Operation
{
    public Weights Queries { get; } = queries;
    public Weights Keys { get; } = keys;
    public override Weights Result { get; } = result;

    private readonly DotProductOperation dotProduct = new(
        new RowReferenceWeights(queries, "queryRowIndex"),
        new RowReferenceWeights(keys, "keyRowIndex"),
        new ItemReferenceWeights(result, ["queryRowIndex", "keyRowIndex"])
    );
    public override void AppendCode(MethodBodyWriter sb)
    {
        Debug.Assert(Queries.Dimensions.SequenceEqual(Keys.Dimensions));

        sb.WriteOperation($"var scale = (Weight)(1 / Weight.Sqrt({Queries.Dimensions[1]}));");
        sb.WriteOperation($"foreach (var queryRowIndex in ..{Queries.Dimensions[0]})");
        sb.OpenScope();
        sb.WriteOperation($"foreach (var keyRowIndex in ..{Queries.Dimensions[0]})");
        sb.OpenScope();
        dotProduct.AppendCode(sb);
        sb.CloseScope();
        sb.CloseScope();
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry, OperationFactory factory)
    {
        dotProduct.AppendGradientOp(ops, registry, factory);
    }
}
