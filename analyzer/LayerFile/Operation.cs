using System.Collections.Generic;
using ML.Analyzer.LayerFile.Operations;

namespace ML.Analyzer.LayerFile;

internal abstract class Operation
{
    public abstract Weights Result { get; }
    public abstract void AppendCode(MethodBodyWriter sb);
    public abstract void AppendGradientOp(List<Operation> ops, LayerRegistry registry, OperationFactory factory);
}
