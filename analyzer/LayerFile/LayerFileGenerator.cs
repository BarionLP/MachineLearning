using System.Collections.Generic;
using System.IO;

namespace ML.Analyzer.LayerFile;

[Generator]
public sealed class LayerFileGenerator : IIncrementalGenerator
{
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        if (!Debugger.IsAttached)
            Debugger.Launch();

        var files = context.AdditionalTextsProvider.Where(a => a.Path.EndsWith(".layer"));

        context.RegisterSourceOutput(files, static (context, file) =>
        {
            var text = file.GetText(context.CancellationToken)!.ToString();

            var layer = LayerFileParser.Parse(Path.GetFileNameWithoutExtension(file.Path), text);
            var registry = layer.Registry;
            var learnedWeights = layer.LearnedWeights;
            var forwardPass = layer.ForwardPass;
            var hasActivationFunction = layer.HasActivationFunction;

            var sb = new StringBuilder();

            sb.AppendLine("using Ametrin.Numerics;");
            sb.AppendLine("using MachineLearning.Model.Layer;");
            sb.AppendLine("using MachineLearning.Model.Layer.Snapshot;");

            sb.AppendLine();

            if (!string.IsNullOrEmpty(layer.Namespace))
            {
                sb.AppendLine($$"""namespace {{layer.Namespace}};""");
            }

            // generate model file


            sb.AppendLine($$"""
            public sealed partial class {{layer.Name}}({{(hasActivationFunction ? "MachineLearning.Model.Activation.IActivationFunction ActivationFunction, " : "")}}{{string.Join(", ", registry.Parameters.Select(p => $"int {p.Name}"))}}) : ILayer<{{layer.Input.Type}}, {{layer.Name}}.Snapshot>
            {
            """);

            if (hasActivationFunction)
            {
                sb.AppendLine($$"""
                    public MachineLearning.Model.Activation.IActivationFunction ActivationFunction { get; } = ActivationFunction; 
                """);
            }

            foreach (var parameter in registry.Parameters)
            {
                sb.AppendLine($$"""
                    public int {{parameter.Access(Location.Layer)}} { get; } = {{parameter.Name}};
                """);
            }

            foreach (var weight in learnedWeights)
            {
                sb.AppendLine($$"""
                    public {{weight.Type}} {{weight.Name}} { get; } = {{weight.Type}}.Create({{string.Join(", ", weight.Dimensions.Select(static p => p.Access(Location.Layer)))}});
                """);
            }


            sb.AppendLine($$"""

                public long WeightCount => {{string.Join(" + ", learnedWeights.Select(static w => w.Type switch { NumberType.Single => "1", NumberType.Vector => $"{w.Access(Location.Layer)}.Count", _ => $"{w.Access(Location.Layer)}.FlatCount" }))}};

                public Vector Forward({{layer.Input.Type}} {{layer.Input.Name}}, Snapshot snapshot)
                {
            """);

            foreach (var operation in forwardPass)
            {
                sb.Append("        ");
                operation.AppendCode(sb);
            }

            sb.AppendLine($$"""
                }
            """);

            sb.AppendLine($$"""
                public Vector Backward({{layer.Input.Type}} outputGradient, Snapshot snapshot, Gradients gradients)
                {
            """);

            foreach (var operation in layer.BackwardPass)
            {
                sb.Append("        ");
                operation.AppendCode(sb);
            }

            sb.AppendLine($$"""
                }
            """);

            sb.AppendLine($$"""

                public Snapshot CreateSnapshot() => new(this);
                ILayerSnapshot MachineLearning.Model.Layer.ILayer.CreateSnapshot() => CreateSnapshot();
                public Gradients CreateGradientAccumulator() => new(this);
                IGradients MachineLearning.Model.Layer.ILayer.CreateGradientAccumulator() => CreateGradientAccumulator();

                public sealed class Snapshot({{layer.Name}} layer) : ILayerSnapshot
                {
            """);

            foreach (var snap in registry.Weights.Distinct().OfType<DirectWeights>().Where(static w => w.Location is Location.Snapshot))
            {
                sb.AppendLine($$"""
                    public {{snap.Type}} {{snap.Name}} { get; } = {{snap.Type}}.Create({{string.Join(", ", snap.Dimensions.Select(static p => p.Access(Location.Snapshot)))}});
            """);
            }

            sb.AppendLine($$"""
                }

                public sealed class Gradients({{layer.Name}} layer) : IGradients
                {
            """);

            var weightsGradientPairs = learnedWeights.Select(w => (w, registry.GetGradient(w)));

            foreach (var (weight, gradient) in weightsGradientPairs)
            {
                sb.AppendLine($$"""
                    public {{gradient.Type}} {{gradient.Name}} { get; } = {{gradient.Type}}.OfSize({{weight.Access(Location.Gradients)}});
            """);
            }

            sb.AppendLine($$"""
                    public void Add(IGradients other)
                    {
                        var o = Ametrin.Guards.Guard.Is<Gradients>(other);
            """);

            foreach (var (weight, gradient) in weightsGradientPairs)
            {
                sb.AppendLine($"            {gradient.Name}.AddToSelf(o.{gradient.Name});");
            }

            sb.AppendLine($$"""
                    }

                    public void Reset()
                    {
            """);

            foreach (var (weight, gradient) in weightsGradientPairs)
            {
                sb.AppendLine($"            {gradient.Name}.ResetZero();");
            }

            sb.AppendLine($$"""
                    }
                }
            """);

            if (layer.Serializer is { } infos)
            {
                sb.Insert(0, "using MachineLearning.Serialization;\n");
                sb.AppendLine($$"""
                public static partial class Serializer
                {
                    [System.Runtime.CompilerServices.ModuleInitializer]
                    internal static void Register()
                    {
                        MachineLearning.Serialization.ModelSerializer.RegisterLayer("{{infos.id}}", {{infos.version}}, Save, Read);
                    }

                    public static ErrorState Save({{layer.Name}} layer, System.IO.BinaryWriter writer)
                    {
                        {{(hasActivationFunction ? "ActivationFunctionSerializer.Write(writer, layer.ActivationFunction);" : "")}}
                        {{string.Join("\n\t\t\t", registry.Parameters.Select(w => $"writer.Write({w.Access(Location.Serializer)});"))}}
                        {{string.Join("\n\t\t\t", learnedWeights.Select(w => $"ModelSerializationHelper.Write{w.Type}({w.Access(Location.Serializer)}, writer);"))}}
                        return default;
                    }

                    public static Result<{{layer.Name}}> Read(System.IO.BinaryReader reader)
                    {
                        var layer = new {{layer.Name}}({{(hasActivationFunction ? $"ActivationFunctionSerializer.Read(reader), " : "")}}{{string.Join(", ", registry.Parameters.Select(static w => $"reader.ReadInt32()"))}});
            """);

                foreach (var weight in learnedWeights)
                {
                    sb.AppendLine($$"""            ModelSerializationHelper.Read{{weight.Type}}(reader).CopyTo({{weight.Access(Location.Serializer)}});""");
                }

                sb.AppendLine($$"""
                        return layer;
                    }
                }
            """);
            }

            sb.AppendLine($$"""
            }
            """);

            context.AddSource($"{layer.Name}.g.cs", sb.ToString());

            AdamLayerGenerator.GenerateAdam(context, new(layer.Name, layer.Namespace, layer.Input.Type.ToString(), layer.Output.Type, $"{layer.Name}.Snapshot", learnedWeights));
        });
    }

    private static IEnumerable<string> GetLines(string text)
    {
        foreach (var l in text.Split('\n'))
        {
            var line = l.Trim();

            if (string.IsNullOrEmpty(line) || line.StartsWith("//")) continue;

            yield return line;
        }
    }
}

internal abstract record Parameter
{
    public abstract string Access(Location location);
}

internal sealed record ValueParameter(int Value) : Parameter
{
    public static ValueParameter Zero { get; } = new(0);
    public override string Access(Location location) => Value.ToString();
}

internal sealed record ReferenceParameter(string Name) : Parameter
{
    public override string Access(Location location)
    {
        if (location is Location.Layer or Location.Pass) return Name;
        return $"layer.{Name}";
    }
}


internal enum Location { Layer, Snapshot, Gradients, Serializer, Pass }
internal enum NumberType { Single, Vector, Matrix, Tensor }
