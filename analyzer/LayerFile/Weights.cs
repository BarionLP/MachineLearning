namespace ML.Analyzer.LayerFile;

internal abstract record Weights(ImmutableArray<Parameter> Dimensions)
{
    public NumberType Type { get; } = Dimensions.Length switch
    {
        0 => NumberType.Single,
        1 => NumberType.Vector,
        2 => NumberType.Matrix,
        3 => NumberType.Tensor,
        _ => throw new InvalidOperationException(),
    };

    public string PassAccess() => Access(Location.Pass);
    public abstract string Access(Location from);
}

internal sealed record DirectWeights(string Name, ImmutableArray<Parameter> Dimensions, Location Location) : Weights(Dimensions)
{
    public override string ToString() => $"{Name} [{string.Join(", ", Dimensions.Select(d => d.Access(Location.Layer)))}]";
    public string GetGradientName() => $"{Name}Gradient";
    public override string Access(Location from)
    {
        if (from == Location) return Name;

        return (from, Location) switch
        {
            (Location.Pass, Location.Layer) => Name,
            (Location.Pass, Location.Snapshot) => $"snapshot.{Name}",
            (Location.Pass, Location.Gradients) => $"gradients.{Name}",
            (Location.Gradients, Location.Layer) => $"layer.{Name}",
            (Location.Serializer, Location.Layer) => $"layer.{Name}",
            _ => throw new InvalidOperationException($"Cannot access {Location} {this} from {from}"),
        };
    }
}

internal sealed record RowReferenceWeights(DirectWeights Matrix, string Row) : Weights(Matrix.Type is NumberType.Matrix ? [Matrix.Dimensions[1]] : throw new InvalidOperationException($"cannot refenrence a row of {Matrix}"))
{
    public override string ToString() => $"{Matrix.Name}[{Row}] [{string.Join(", ", Dimensions.Select(d => d.Access(Location.Layer)))}]";

    public override string Access(Location from) => $"{Matrix.Access(from)}.RowRef({Row})";
}

internal sealed record ItemReferenceWeights(DirectWeights Weights, ImmutableArray<string> Accessor) : Weights(Weights.Dimensions.Length == Accessor.Length ? [] : throw new InvalidOperationException($"cannot refenrence a value of {Weights} with [{string.Join(", ", Accessor)}]"))
{
    public override string ToString() => $"{Weights.Name}[{string.Join(", ", Accessor)}]";

    public override string Access(Location from) => $"{Weights.Access(from)}[{string.Join(", ", Accessor)}]";
}