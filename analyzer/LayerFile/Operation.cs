using System.Collections.Generic;

namespace ML.Analyzer.LayerFile;

internal abstract class Operation
{
    public abstract Weights Result { get; }
    public abstract void AppendCode(StringBuilder sb);
    public abstract void AppendGradientOp(List<Operation> ops, LayerRegistry registry);
}
