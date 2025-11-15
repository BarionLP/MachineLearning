using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;

namespace ML.Analyzer.LayerFile;

internal sealed class LayerRegistry
{
    private readonly Dictionary<string, Weights> weightsLookup = [];
    private readonly Dictionary<string, ReferenceParameter> paramLookup = [];

    public Weights this[string name, [CallerFilePath] string file = default!, [CallerLineNumber] int line = -1]
    {
        get
        {
            if (name[^1] is ']')
            {
                var idx = name.IndexOf('[');
                var accessor = name.Substring(idx + 1, name.Length - idx - 2).Split([", "], StringSplitOptions.None).Select(static s => s.Trim()).ToImmutableArray();
                var weights = (DirectWeights)this[name.Substring(0, idx)];
                return weights.Type is NumberType.Matrix && accessor.Length is 1
                    ? new RowReferenceWeights(weights, accessor[0])
                    : new ItemReferenceWeights(weights, accessor);
            }
            return weightsLookup.TryGetValue(name, out var value) ? value : throw new InvalidOperationException($"{name} requested at {Path.GetFileNameWithoutExtension(file)}:{line} is not defined! {string.Join(" ", Weights)}");
        }
    }

    public Dictionary<string, Weights>.ValueCollection Weights => weightsLookup.Values;
    public Dictionary<string, ReferenceParameter>.ValueCollection Parameters => paramLookup.Values;

    public Weights GetOrCreateGradient(Weights original) => TryGetGradient(original) ?? CreateWeightsGradient(original);
    public Weights GetGradient(Weights original) => TryGetGradient(original) ?? throw new InvalidOperationException($"Gradient for {original} not created");
    public Weights? TryGetGradient(Weights original)
    {
        return original switch
        {
            DirectWeights dw => TryGetGradient(dw),
            RowReferenceWeights rw => TryGetGradient(rw.Matrix) is { } w ? rw with { Matrix = w } : null,
            ItemReferenceWeights rw => TryGetGradient(rw.Weights) is { } w ? rw with { Weights = w } : null,
            ConditionalReferenceWeights cw => TryGetGradient(cw.WhenTrue) is { } w ? cw with { WhenTrue = w, WhenFalse = null } : null,
            _ => throw new InvalidOperationException($"unkown weight type {original}"),
        };
    }

    public DirectWeights GetGradient(DirectWeights original) => TryGetGradient(original) ?? throw new InvalidOperationException($"Gradient for {original} not created");
    public DirectWeights? TryGetGradient(DirectWeights original) => weightsLookup.TryGetValue(original.GetGradientName(), out var value) ? (DirectWeights?)(DirectWeights)value : null;

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

    public DirectWeights CreateWeights(string name, ImmutableArray<Parameter> dimensions, Location location, bool readOnlyProperty = true)
    {
        ThrowIfDuplicate(name);

        var obj = new DirectWeights(name, dimensions, location, readOnlyProperty);
        weightsLookup.Add(name, obj);
        return obj;
    }

    public Weights CreateWeightsGradient(Weights original) => original switch
    {
        DirectWeights dw => CreateWeightsGradient(dw),
        RowReferenceWeights rw => rw with { Matrix = CreateWeightsGradient(rw.Matrix) },
        ItemReferenceWeights rw => rw with { Weights = CreateWeightsGradient(rw.Weights) },
        ConditionalReferenceWeights cw => cw with { WhenTrue = CreateWeightsGradient(cw.WhenTrue), WhenFalse = null },
        _ => throw new InvalidOperationException($"unkown weight type {original}"),
    };
    public DirectWeights CreateWeightsGradient(DirectWeights original) => CreateWeightsGradient(original, original.Location switch
    {
        Location.Layer => Location.Gradients,
        Location.Snapshot => Location.Snapshot,
        _ => throw new InvalidOperationException($"unkown location type {original.Location}")
    });

    public Weights CreateWeightsGradient(Weights original, Location location) => original switch
    {
        DirectWeights dw => CreateWeightsGradient(dw, location),
        RowReferenceWeights rw => rw with { Matrix = CreateWeightsGradient(rw.Matrix, location) },
        ItemReferenceWeights rw => rw with { Weights = CreateWeightsGradient(rw.Weights, location) },
        ConditionalReferenceWeights cw => cw with { WhenTrue = CreateWeightsGradient(cw.WhenTrue, location), WhenFalse = null },
        _ => throw new InvalidOperationException($"unkown weight type {original}"),
    };
    public DirectWeights CreateWeightsGradient(DirectWeights original, Location location)
    {
        var name = original.GetGradientName();

        ThrowIfDuplicate(name);

        var obj = new DirectWeights(name, original.Dimensions, location);
        weightsLookup.Add(name, obj);
        return obj;
    }

    public DirectWeights ParseWeightDefinition(ReadOnlySpan<char> line, Location location, bool readOnlyProperty = true)
    {
        var nameEndIndex = line.IndexOf('[');
        var name = line.Slice(0, nameEndIndex).Trim().ToString();

        var dimensionsSpan = line.Slice(nameEndIndex + 1, line.IndexOf(']') - nameEndIndex - 1).Trim();
        var dimensions = dimensionsSpan.ToString().Split(',').Select(static s => s.Trim()).Select<string, Parameter>(s => int.TryParse(s, out var value) ? new ValueParameter(value) : paramLookup.TryGetValue(s, out var reference) ? reference : throw new InvalidOperationException($"Undefined Parameter {s}"));

        return CreateWeights(name, [.. dimensions], location, readOnlyProperty);
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