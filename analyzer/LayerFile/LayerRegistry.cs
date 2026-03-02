using System.Collections.Generic;
using System.IO;
using System.Runtime.CompilerServices;

namespace ML.Analyzer.LayerFile;

internal sealed class LayerRegistry
{
    private readonly Dictionary<string, Weights> weightsLookup = [];
    private readonly Dictionary<string, ReferenceParameter> paramLookup = [];
    internal readonly Dictionary<string, Module> moduleLookup = [];

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

    public Weights GetOrCreateGradient(Weights original, bool preAllocate = true) => TryGetGradient(original) ?? CreateWeightsGradient(original, preAllocate);
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
    public DirectWeights? TryGetGradient(DirectWeights original) => weightsLookup.TryGetValue(original.GetGradientName(), out var value) ? (DirectWeights?)value : null;

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

    public Weights CreateWeightsGradient(Weights original, bool preAllocate = true) => original switch
    {
        DirectWeights dw => CreateWeightsGradient(dw, preAllocate),
        RowReferenceWeights rw => rw with { Matrix = CreateWeightsGradient(rw.Matrix, preAllocate) },
        ItemReferenceWeights rw => rw with { Weights = CreateWeightsGradient(rw.Weights, preAllocate) },
        ConditionalReferenceWeights cw => cw with { WhenTrue = CreateWeightsGradient(cw.WhenTrue, preAllocate), WhenFalse = null },
        _ => throw new InvalidOperationException($"unkown weight type {original}"),
    };

    public DirectWeights CreateWeightsGradient(DirectWeights original, bool preAllocate = true) => CreateWeightsGradient(original, original.Location switch
    {
        Location.Layer => Location.Gradients,
        Location.Snapshot => Location.Snapshot,
        _ => throw new InvalidOperationException($"unkown location type {original.Location}")
    }, preAllocate);

    public Weights CreateWeightsGradient(Weights original, Location location, bool preAllocate = true) => original switch
    {
        DirectWeights dw => CreateWeightsGradient(dw, location, preAllocate),
        RowReferenceWeights rw => rw with { Matrix = CreateWeightsGradient(rw.Matrix, location, preAllocate) },
        ItemReferenceWeights rw => rw with { Weights = CreateWeightsGradient(rw.Weights, location, preAllocate) },
        ConditionalReferenceWeights cw => cw with { WhenTrue = CreateWeightsGradient(cw.WhenTrue, location, preAllocate), WhenFalse = null },
        _ => throw new InvalidOperationException($"unkown weight type {original}"),
    };
    public DirectWeights CreateWeightsGradient(DirectWeights original, Location location, bool preAllocate = true)
    {
        var name = original.GetGradientName();

        ThrowIfDuplicate(name);

        var obj = new DirectWeights(name, original.Dimensions, location, preAllocate);
        weightsLookup.Add(name, obj);
        return obj;
    }


    public Weights CreateWeightsGradient(Weights original, Weights reference) => (original, reference) switch
    {
        (DirectWeights dw, DirectWeights) => CreateWeightsGradient(dw, reference),
        (RowReferenceWeights rw, RowReferenceWeights rrw) => rw with { Matrix = CreateWeightsGradient(rw.Matrix, rrw.Matrix) },
        (ItemReferenceWeights rw, ItemReferenceWeights rrw) => rw with { Weights = CreateWeightsGradient(rw.Weights, rrw.Weights) },
        (ConditionalReferenceWeights cw, ConditionalReferenceWeights rcw) => cw with { WhenTrue = CreateWeightsGradient(cw.WhenTrue, rcw.WhenTrue), WhenFalse = null },
        _ => throw new InvalidOperationException($"cannot use {reference} as gradient for {original}"),
    };
    public Weights CreateWeightsGradient(DirectWeights original, Weights reference)
    {
        var name = original.GetGradientName();
        AddAlias(name, reference);
        return reference;
    }

    public DirectWeights ParseWeightDefinition(ReadOnlySpan<char> line, Location location, bool preAllocate = true)
    {
        var nameEndIndex = line.IndexOf('[');
        var name = line.Slice(0, nameEndIndex).Trim().ToString();

        var dimensionsSpan = line.Slice(nameEndIndex + 1, line.IndexOf(']') - nameEndIndex - 1).Trim();
        var rawDimensions = dimensionsSpan.ToString().Split(',').Select(static s => s.Trim()).ToArray();
        (var dimensions, preAllocate) = rawDimensions switch
        {
            [""] => ([new ValueParameter(1)], false),
            ["", ""] => ([new ValueParameter(1), new ValueParameter(1)], false),
            ["", "", ""] => ([new ValueParameter(1), new ValueParameter(1), new ValueParameter(1)], false),
            _ => (rawDimensions.Select<string, Parameter>(s => int.TryParse(s, out var value) ? new ValueParameter(value) : paramLookup.TryGetValue(s, out var reference) ? reference : throw new InvalidOperationException($"Undefined Parameter {s}")), preAllocate),
        };

        return CreateWeights(name, [.. dimensions], location, preAllocate);
    }

    public Module ParseModule(ReadOnlySpan<char> line)
    {
        var parts = line.ToString().Split([' '], StringSplitOptions.RemoveEmptyEntries);

        if (parts.Length < 2)
        {
            throw new InvalidOperationException($"invalid module definition: {line.ToString()}");
        }

        var module = new Module(parts[0], parts[1], [..parts.AsSpan(2)]);

        ThrowIfDuplicate(module.Name);
        moduleLookup.Add(module.Name, module);
        return module;
    }

    public static string GetAccessString(Location from, Location target, string name)
    {
        if (from == target) return name;

        return (from, target) switch
        {
            (Location.Pass, Location.Layer) => name,
            (Location.Pass, Location.Snapshot) => $"snapshot.{name}",
            (Location.Pass, Location.Gradients) => $"gradients.{name}",
            (Location.Gradients, Location.Layer) => $"layer.{name}",
            (Location.Snapshot, Location.Layer) => $"layer.{name}",
            (Location.Serializer, Location.Layer) => $"layer.{name}",
            _ => throw new InvalidOperationException($"Cannot access {target} {name} from {from}"),
        };
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
        if (moduleLookup.TryGetValue(name, out var existingM))
        {
            throw new InvalidOperationException($"{name} is already defined as module {existingM}");
        }
    }
}