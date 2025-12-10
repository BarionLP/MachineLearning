namespace ML.Analyzer.LayerFile.Operations;

internal sealed class OperationFactory(LayerRegistry registry)
{
    private readonly LayerRegistry registry = registry;

    public Operation NewPointwiseMultiply(Weights left, Weights right, Weights result, bool? add = null)
    {
        return CreateConditionalAware(result, result => new PointwiseMultiplyOperation(left, right, result, add));
    }

    public Operation NewMultiply(Weights left, Weights right, Weights result, bool? add = null)
    {
        return CreateConditionalAware(result, result => (left.Type, right.Type) switch
            {
                (NumberType.Matrix, NumberType.Vector) when right is RowReferenceWeights rw => new MatrixVectorMultiplyOperation(left, right, result, add),
                (NumberType.Matrix, NumberType.Vector) => new MatrixVectorMultiplyOperation(left, right, result, add),
                (NumberType.Vector, NumberType.Single) => new VectorSingleMultiplyOperation(left, right, result, add),
                _ => throw new NotImplementedException($"cannot multiply{(add is true ? "-add" : "")} {left} and {right}"),
            }
        );
    }

    public Operation CreateAttention(Weights queries, Weights keys, Weights result)
    {
        return CreateConditionalAware(result, result => new ConstructAttentionMatrixOperation(queries, keys, result));
    }

    private static Operation CreateConditionalAware(Weights relevant, Func<Weights, Operation> factory)
    {
        if (relevant is ConditionalReferenceWeights { Condition: { } condition, WhenTrue: { } inner, WhenFalse: null })
        {
            return new ConditionOperation(condition, factory(inner));
        }
        return factory(relevant);
    }

    public RowwiseRecurrenceOperation CreateRecurrence(ImmutableArray<Weights> weights, bool reversed)
    {
        var currWeights = weights.Cast<DirectWeights>()
            .Select(w =>
            {
                Weights refe = w.Dimensions switch
                {
                    [var v] => new ItemReferenceWeights(w, ["t"]),
                    [var r, var c] => new RowReferenceWeights(w, "t"),
                    _ => throw new InvalidOperationException($"Cannot recur over {w}"),
                };

                registry.AddAlias($"{w.Name}_current", refe);
                return refe;
            });

        var prevWeights = weights.Cast<DirectWeights>()
            .Select(w =>
            {
                Weights refe = w.Dimensions switch
                {
                    [var v] => new ConditionalReferenceWeights("t > 0", new ItemReferenceWeights(w, ["t-1"]), "ZERO"),
                    [var r, var c] => new ConditionalReferenceWeights("t > 0", new RowReferenceWeights(w, "t-1"), "ZERO"),
                    _ => throw new InvalidOperationException($"Cannot recur over {w}"),
                };

                registry.AddAlias($"{w.Name}_previous", refe);
                return refe;
            });
        var recurrence = new RowwiseRecurrenceOperation(weights, [.. currWeights, .. prevWeights], reversed);
        return recurrence;
    }
}
