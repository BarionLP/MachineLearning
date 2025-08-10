using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;

namespace ML.Analyzer.LayerFile;

internal sealed class LayerRegistry
{
    private readonly Dictionary<string, Weights> weightsLookup = [];
    private readonly Dictionary<string, ReferenceParameter> paramLookup = [];

    public Weights this[string name, [CallerFilePath] string file = default!, [CallerLineNumber] int line = -1] => weightsLookup.TryGetValue(name, out var value) ? value : throw new InvalidOperationException($"{name} requested at {Path.GetFileNameWithoutExtension(file)}:{line} is not defined");
    public Dictionary<string, Weights>.ValueCollection Weights => weightsLookup.Values;
    public Dictionary<string, ReferenceParameter>.ValueCollection Parameters => paramLookup.Values;

    public Weights GetGradient(Weights original) => weightsLookup.TryGetValue(original.GetGradientName(), out var value) ? value : throw new InvalidOperationException($"Gradient for {original.Name} not created");

    public void AddAlias(string name, Weights weights)
    {
        ThrowIfDuplicate(name);
        weightsLookup.Add(name, weights);
    }

    public void CreateParameter(string name)
    {
        ThrowIfDuplicate(name);
        paramLookup.Add(name, new(name));
    }

    public Weights CreateWeights(string name, ImmutableArray<Parameter> dimensions, Location location)
    {
        ThrowIfDuplicate(name);

        var obj = new Weights(name, dimensions, location);
        weightsLookup.Add(name, obj);
        return obj;
    }

    public Weights CreateWeightsGradient(Weights original) => CreateWeightsGradient(original, original.Location switch
    {
        Location.Layer => Location.Gradients,
        Location.Snapshot => Location.Snapshot,
        _ => throw new InvalidOperationException()
    });

    public Weights CreateWeightsGradient(Weights original, Location location)
    {
        var name = original.GetGradientName();

        ThrowIfDuplicate(name);

        var obj = new Weights(name, original.Dimensions, location);
        weightsLookup.Add(name, obj);
        return obj;
    }

    public Weights ParseWeightDefinition(ReadOnlySpan<char> line, Location location)
    {
        var nameEndIndex = line.IndexOf('[');
        var name = line.Slice(0, nameEndIndex).Trim().ToString();

        ThrowIfDuplicate(name);

        var dimensionsSpan = line.Slice(nameEndIndex + 1, line.IndexOf(']') - nameEndIndex - 1).Trim();
        var dimensions = dimensionsSpan.ToString().Split(',').Select(static s => s.Trim()).Select<string, Parameter>(s => int.TryParse(s, out var value) ? new ValueParameter(value) : paramLookup.TryGetValue(s, out var reference) ? reference : throw new InvalidOperationException($"Undefined Parameter {s}"));

        var obj = new Weights(name, [.. dimensions], location);
        weightsLookup.Add(name, obj);
        return obj;
    }

    private void ThrowIfDuplicate(string name)
    {
        if (weightsLookup.TryGetValue(name, out var existingW))
        {
            throw new InvalidOperationException($"{name} is already defined as weight {existingW}");
        }
        if (paramLookup.TryGetValue(name, out var existingP))
        {
            throw new InvalidOperationException($"{name} is already defined as parameter {existingP}");
        }
    }
}