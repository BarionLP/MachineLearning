using System.Collections.Generic;
using System.IO;
using ML.Analyzer.LayerFile.Operations;

namespace ML.Analyzer.LayerFile;

[Generator]
public sealed class LayerFileGenerator : IIncrementalGenerator
{
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var files = context.AdditionalTextsProvider.Where(a => a.Path.EndsWith(".layer"));

        context.RegisterSourceOutput(files, static (context, file) =>
        {
            var registry = new LayerRegistry();
            var learnedWeights = new List<Weights>();
            var forwardPass = new List<Operation>();
            var hasActivationFunction = false;

            var text = file.GetText(context.CancellationToken)!.ToString();
            var lines = GetLines(text).GetEnumerator();

            var sb = new StringBuilder();

            sb.AppendLine("using Ametrin.Numerics;");
            sb.AppendLine("using MachineLearning.Model.Layer;");
            sb.AppendLine("using MachineLearning.Model.Layer.Snapshot;");

            sb.AppendLine();

            lines.MoveNext();

            var modelNamespace = string.Empty;
            if (lines.Current.StartsWith("# namespace "))
            {
                modelNamespace = lines.Current.Substring("# namespace ".Length).Trim();
                sb.AppendLine($"namespace {modelNamespace};");
                sb.AppendLine();
                lines.MoveNext();
            }

            var layerName = Path.GetFileNameWithoutExtension(file.Path);

            if (lines.Current is "# Activation Function")
            {
                hasActivationFunction = true;
                lines.MoveNext();
            }

            if (lines.Current is "# Parameters")
            {
                while (lines.MoveNext())
                {
                    if (lines.Current is "# Weights") break;

                    registry.CreateParameter(lines.Current);
                }
            }

            if (lines.Current is not "# Weights") return;

            // list of weights
            while (lines.MoveNext())
            {
                if (lines.Current.StartsWith("# Forward ")) break;

                learnedWeights.Add(registry.ParseWeightDefinition(lines.Current.AsSpan().TrimEnd(), Location.Layer));
            }

            var inputWeights = registry.ParseWeightDefinition(lines.Current.AsSpan("# Forward".Length).Trim(), Location.Snapshot);
            forwardPass.Add(new InputOperation(inputWeights with { Location = Location.Pass }, inputWeights));

            var factory = new OperationFactory(registry);

            // forward pass
            while (lines.MoveNext())
            {
                var parts = lines.Current.Split(' ');
                forwardPass.Add(parts switch
                {
                    [var result, "=", var source, "Activate"] => factory.NewLeakyReLU(registry[source], result),
                    [var result, "=", var left, "+", var right] => factory.NewAdd(registry[left], registry[right], result),
                    [var result, "=", var left, "*", var right] => factory.NewMatrixVectorMultiply(registry[left], registry[right], result),
                    [var output] => new OutputOperation(registry[output]),
                    _ => throw new InvalidOperationException($"unkown operation {lines.Current}"),
                });

                if (forwardPass[^1] is OutputOperation) break;
            }


            var outputWeights = forwardPass[^1].Result;


            // construct backwards pass
            var backwardsPass = new List<Operation>();
            forwardPass.Reverse();
            foreach (var operation in forwardPass)
            {
                operation.AppendGradientOp(backwardsPass, registry);
            }
            forwardPass.Reverse();



            (string id, int version)? serializerInfos = null;
            lines.MoveNext();
            if (lines.Current.StartsWith("# Serializer "))
            {
                var parts = lines.Current.Split(' ');
                serializerInfos = (parts[2], int.Parse(parts[3]));
            }

            // generate model file

            sb.AppendLine($$"""
            public sealed partial class {{layerName}}({{(hasActivationFunction ? "MachineLearning.Model.Activation.IActivationFunction ActivationFunction, " : "")}}{{string.Join(", ", registry.Parameters.Select(p => $"int {p.Name}"))}}) : ILayer<{{inputWeights.Type}}, {{layerName}}.Snapshot>
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

                public Vector Forward({{inputWeights.Type}} {{inputWeights.Name}}, Snapshot snapshot)
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
                public Vector Backward({{inputWeights.Type}} outputGradient, Snapshot snapshot, Gradients gradients)
                {
            """);

            foreach (var operation in backwardsPass)
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

                public sealed class Snapshot({{layerName}} layer) : ILayerSnapshot
                {
            """);

            foreach (var snap in registry.Weights.Distinct().Where(static w => w.Location is Location.Snapshot))
            {
                sb.AppendLine($$"""
                    public {{snap.Type}} {{snap.Name}} { get; } = {{snap.Type}}.Create({{string.Join(", ", snap.Dimensions.Select(static p => p.Access(Location.Snapshot)))}});
            """);
            }

            sb.AppendLine($$"""
                }

                public sealed class Gradients({{layerName}} layer) : IGradients
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

            if (serializerInfos is { } infos)
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

                    public static ErrorState Save({{layerName}} layer, System.IO.BinaryWriter writer)
                    {
                        {{(hasActivationFunction ? "ActivationFunctionSerializer.Write(writer, layer.ActivationFunction);" : "")}}
                        {{string.Join("\n\t\t\t", registry.Parameters.Select(w => $"writer.Write({w.Access(Location.Serializer)});"))}}
                        {{string.Join("\n\t\t\t", learnedWeights.Select(w => $"ModelSerializationHelper.Write{w.Type}({w.Access(Location.Serializer)}, writer);"))}}
                        return default;
                    }

                    public static Result<{{layerName}}> Read(System.IO.BinaryReader reader)
                    {
                        var layer = new {{layerName}}({{(hasActivationFunction ? $"ActivationFunctionSerializer.Read(reader), " : "")}}{{string.Join(", ", registry.Parameters.Select(static w => $"reader.ReadInt32()"))}});
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

            context.AddSource($"{layerName}.g.cs", sb.ToString());

            AdamLayerGenerator.GenerateAdam(context, new(layerName, modelNamespace, inputWeights.Type.ToString(), outputWeights.Type, $"{layerName}.Snapshot", learnedWeights));
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
