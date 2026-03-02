using System.Collections.Generic;
using ML.Analyzer.LayerFile.Operations;

namespace ML.Analyzer.LayerFile;

internal static class LayerFileParser
{
    public static LayerDefinition Parse(string name, string text)
    {
        var def = new LayerDefinition
        {
            Name = name,
        };

        var lines = new LineEnumerator(text);

        lines.MoveNext();

        // namespace
        if (lines.Current.StartsWith("# namespace ".AsSpan()))
        {
            def.Namespace = lines.Current.Slice("# namespace".Length).Trim().ToString();
            lines.MoveNext();
        }

        // activation function
        while (lines.Current.StartsWith("# Activation Function"))
        {
            def.ActivationFunctions.Add(lines.Current.Slice("# Activation Function".Length).Trim().ToString());
            lines.MoveNext();
        }

        // parameters
        if (lines.Current is "# Parameters")
        {
            while (lines.MoveNext())
            {
                if (lines.Current is "# Weights" or "# Modules") break;

                def.Registry.CreateParameter(lines.Current.ToString());
            }
        }

        if (lines.Current is "# Modules")
        {
            while (lines.MoveNext())
            {
                if (lines.Current is "# Weights") break;

                def.Modules.Add(def.Registry.ParseModule(lines.Current));
            }
        }

        // weights
        if (lines.Current is not "# Weights") throw new InvalidOperationException($"{name} has no weights");

        while (lines.MoveNext())
        {
            if (lines.Current.StartsWith("# Forward ") || lines.Current is "# Snapshot") break;

            def.LearnedWeights.Add(def.Registry.ParseWeightDefinition(lines.Current, Location.Layer));
        }

        // snapshot
        if (lines.Current is "# Snapshot")
        {
            while (lines.MoveNext())
            {
                if (lines.Current.StartsWith("# Forward ")) break;

                def.Registry.ParseWeightDefinition(lines.Current, Location.Snapshot);
            }
        }

        def.Input = def.Registry.ParseWeightDefinition(lines.Current.Slice("# Forward".Length).Trim(), Location.Snapshot, preAllocate: false);
        def.ForwardPass.Add(new InputOperation(def.Input.WithLocation(Location.Pass), def.Input));

        var factory = new OperationFactory(def.Registry);

        var contextStack = new Stack<RowwiseRecurrenceOperation>();

        // forward pass
        while (lines.MoveNext())
        {
            var parts = lines.Current.ToString().Split(' ');
            def.ForwardPass.Add(parts switch
            {
                [var result, "=", "activate", var activationFunction, var source] => def.ActivationFunctions.Contains(activationFunction) ? new ActivationOperation(def.Registry[source], def.Registry[result], activationFunction) : throw new InvalidOperationException($"activation function {activationFunction} not defined"),
                [var result, "=", var left, "+", var right] => new AddOperation(def.Registry[left], def.Registry[right], def.Registry[result]),
                [var result, "+=", var other] => new AddOperation(def.Registry[result], def.Registry[other], def.Registry[result]),
                [var result, "=", var left, "*", var right] => factory.NewMultiply(def.Registry[left], def.Registry[right], def.Registry[result], add: false),
                [var result, "+=", var left, "*", var right] => factory.NewMultiply(def.Registry[left], def.Registry[right], def.Registry[result], add: true),
                [var result, "=", var left, "⊙", var right] => factory.NewPointwiseMultiply(def.Registry[left], def.Registry[right], def.Registry[result], add: false),
                [var result, "+=", var left, "⊙", var right] => factory.NewPointwiseMultiply(def.Registry[left], def.Registry[right], def.Registry[result], add: true),
                [var result, "=", var queries, "attend", var keys] => factory.CreateAttention(def.Registry[queries], def.Registry[keys], def.Registry[result]),
                [var result, "=", "softmax", var input] => new ActivationOperation(def.Registry[input], def.Registry[result], "SoftMaxActivation.Instance"),
                [var result, "=", var module, "forward", var input] => new NestedLayerOperation(def.Registry.moduleLookup[module], def.Registry[input], def.Registry[result]),
                ["recur", "over", ..] => factory.CreateRecurrence([.. parts.Skip(2).Select(p => def.Registry[p])], reversed: false),
                ["end"] => new EndLoopOperation(contextStack.Pop()),
                [var output] => new OutputOperation(def.Registry[output]),
                _ => throw new InvalidOperationException($"unkown operation '{lines.Current.ToString()}'"),
            });

            if (def.ForwardPass[^1] is RowwiseRecurrenceOperation lop)
            {
                contextStack.Push(lop);
            }
            if (def.ForwardPass[^1] is OutputOperation) break;
        }

        def.Output = def.ForwardPass[^1].Result;

        // serializer
        lines.MoveNext();
        if (lines.Current.StartsWith("# Serializer ".AsSpan()))
        {
            var parts = lines.Current.ToString().Split(' ');
            def.Serializer = (parts[2], int.Parse(parts[3]));
        }

        // generate backward pass
        def.ForwardPass.Reverse();
        foreach (var operation in def.ForwardPass)
        {
            operation.AppendGradientOp(def.BackwardPass, def.Registry, factory);
        }
        def.ForwardPass.Reverse();


        return def;
    }

    private struct LineEnumerator(string text)
    {
        private readonly string text = text;
        private int start = -1;
        private int length = -1;
        private int nextStart = 0;
        public readonly ReadOnlySpan<char> Current => text.AsSpan(start, length);

        public bool MoveNext()
        {
            if (nextStart >= text.Length) return false;
            var rest = text.AsSpan(nextStart);
            var lineBreakIndex = rest.IndexOf('\n');
            if (lineBreakIndex == -1) lineBreakIndex = rest.Length;

            var line = rest.Slice(0, lineBreakIndex);
            var startTrimmed = line.TrimStart();
            var endTrimmed = line.TrimEnd();

            start = nextStart + (line.Length - startTrimmed.Length);
            length = lineBreakIndex - (line.Length - endTrimmed.Length) - (line.Length - startTrimmed.Length);
            nextStart += lineBreakIndex + 1;

            if (length == -1 || Current.IsEmpty)
            {
                return MoveNext();
            }

            return true;
        }
    }
}

internal sealed class LayerDefinition
{
    public string Namespace { get; set; } = string.Empty;
    public required string Name { get; init; }
    public HashSet<string> ActivationFunctions { get; } = [];
    public LayerRegistry Registry { get; } = new();
    public List<DirectWeights> LearnedWeights { get; } = [];
    public List<Operation> ForwardPass { get; } = [];
    public List<Operation> BackwardPass { get; } = [];
    public List<Module> Modules { get; } = [];
    public DirectWeights Input { get; set; } = default!;
    public Weights Output { get; set; } = default!;
    public (string id, int version)? Serializer { get; set; } = null;
}


internal sealed record Module(string Type, string Name, ImmutableArray<string>? Args)
{
    public string Access(Location from) => LayerRegistry.GetAccessString(from, Location.Layer, Name);
    public string AccessSnapshot(Location from) => LayerRegistry.GetAccessString(from, Location.Snapshot, Name);
    public string AccessGradients(Location from) => LayerRegistry.GetAccessString(from, Location.Gradients, Name);
}