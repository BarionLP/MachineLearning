namespace ML.Analyzer.LayerFile;

internal sealed record Weights(string Name, ImmutableArray<Parameter> Dimensions, Location Location)
{
    public NumberType Type { get; } = Dimensions.Length switch
    {
        0 => NumberType.Single,
        1 => NumberType.Vector,
        2 => NumberType.Matrix,
        3 => NumberType.Tensor,
        _ => throw new InvalidOperationException(),
    };

    public override string ToString() => $"{Name} {string.Join(", ", Dimensions.Select(d => d.Access(Location.Layer)))}";

    public string PassAccess() => Access(Location.Pass);
    public string Access(Location from)
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
    public string GetGradientName() => $"{Name}Gradient";
}
