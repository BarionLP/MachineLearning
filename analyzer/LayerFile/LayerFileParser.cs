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

        var lines = new LineIterator(text);

        lines.MoveNext();

        // namespace
        if (lines.Current.StartsWith("# namespace ".AsSpan()))
        {
            def.Namespace = lines.Current.Slice("# namespace".Length).Trim().ToString();
            lines.MoveNext();
        }

        // activation function
        if (lines.Current is "# Activation Function")
        {
            def.HasActivationFunction = true;
            lines.MoveNext();
        }

        // parameters
        if (lines.Current is "# Parameters")
        {
            while (lines.MoveNext())
            {
                if (lines.Current is "# Weights") break;

                def.Registry.CreateParameter(lines.Current.ToString());
            }
        }

        // weights
        if (lines.Current is not "# Weights") throw new InvalidOperationException($"{name} has no weights");

        while (lines.MoveNext())
        {
            if (lines.Current.StartsWith("# Forward ".AsSpan())) break;

            def.LearnedWeights.Add(def.Registry.ParseWeightDefinition(lines.Current, Location.Layer));
        }

        def.Input = def.Registry.ParseWeightDefinition(lines.Current.Slice("# Forward".Length).Trim(), Location.Snapshot);
        def.ForwardPass.Add(new InputOperation(def.Input with { Location = Location.Pass }, def.Input));

        var factory = new OperationFactory(def.Registry);

        // forward pass
        while (lines.MoveNext())
        {
            var parts = lines.Current.ToString().Split(' ');
            def.ForwardPass.Add(parts switch
            {
                [var result, "=", var source, "Activate"] => factory.NewLeakyReLU(def.Registry[source], result),
                [var result, "=", var left, "+", var right] => factory.NewAdd(def.Registry[left], def.Registry[right], result),
                [var result, "=", var left, "*", var right] => factory.NewMatrixVectorMultiply(def.Registry[left], def.Registry[right], result),
                [var output] => new OutputOperation(def.Registry[output]),
                _ => throw new InvalidOperationException($"unkown operation {lines.Current.ToString()}"),
            });

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
            operation.AppendGradientOp(def.BackwardPass, def.Registry);
        }
        def.ForwardPass.Reverse();


        return def;
    }

    private struct LineIterator(string text)
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

            if (Current.IsEmpty)
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
    public bool HasActivationFunction { get; set; } = false;
    public LayerRegistry Registry { get; } = new();
    public List<Weights> LearnedWeights { get; } = [];
    public List<Operation> ForwardPass { get; } = [];
    public List<Operation> BackwardPass { get; } = [];
    public Weights Input { get; set; } = default!;
    public Weights Output { get; set; } = default!;
    public (string id, int version)? Serializer { get; set; } = null;
}
