using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class ForeachRowOperation(Weights matrix, string indexName, bool reversed) : Operation
{
    public Weights Matrix { get; } = matrix.Type is NumberType.Matrix ? matrix : throw new InvalidOperationException("Can only enumerate rows on a matrix");
    public override Weights Result { get; } = new DirectWeights(indexName, [], Location.Pass);
    public bool Reversed { get; } = reversed;

    public override void AppendCode(StringBuilder sb)
    {
        sb.AppendLine(Reversed
            ? $$"""for(int {{((DirectWeights)Result).Name}} = {{Matrix.PassAccess()}}.RowCount - 1; i <= 0 , i++) {"""
            : $$"""for(int {{((DirectWeights)Result).Name}} = 0; i < {{Matrix.PassAccess()}}.RowCount, i++) {"""
        );
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry)
    {
        ops.Add(new EndLoopOperation(this));
    }
}

internal sealed class EndLoopOperation(ForeachRowOperation loop) : Operation
{
    public ForeachRowOperation Loop { get; } = loop;
    public override Weights Result => null!;


    public override void AppendCode(StringBuilder sb)
    {
        sb.AppendLine("}");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry)
    {
        ops.Add(new ForeachRowOperation(Loop.Matrix, ((DirectWeights)Loop.Result).Name, !Loop.Reversed));
    }
}