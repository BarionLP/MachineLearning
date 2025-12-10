using System.Collections.Generic;

namespace ML.Analyzer.LayerFile.Operations;

internal sealed class RowwiseRecurrenceOperation(ImmutableArray<Weights> weights, ImmutableArray<Weights> tempWeights, bool reversed) : Operation
{
    public ImmutableArray<Weights> Weights { get; } = weights;
    public ImmutableArray<Weights> TempWeights { get; } = tempWeights;
    public override Weights Result { get; } = null!;
    public bool Reversed { get; } = reversed;

    public override void AppendCode(MethodBodyWriter sb)
    {
        sb.WriteOperation(Reversed
            ? $$"""for(int t = {{Weights[0].PassAccess()}}.RowCount - 1; t >= 0; t--)"""
            : $$"""for(int t = 0; t < {{Weights[0].PassAccess()}}.RowCount; t++)"""
        );
        sb.WriteOperation("{");
        sb.Indent++;
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry, OperationFactory factory)
    {
        ops.Add(new EndLoopOperation(this));
    }
}

internal sealed class EndLoopOperation(RowwiseRecurrenceOperation loop) : Operation
{
    public RowwiseRecurrenceOperation Loop { get; } = loop;
    public override Weights Result => null!;


    public override void AppendCode(MethodBodyWriter sb)
    {
        sb.Indent--;
        sb.WriteOperation("}");
    }

    public override void AppendGradientOp(List<Operation> ops, LayerRegistry registry, OperationFactory factory)
    {
        ops.Add(new RowwiseRecurrenceOperation(Loop.Weights, Loop.TempWeights, !Loop.Reversed));
    }
}