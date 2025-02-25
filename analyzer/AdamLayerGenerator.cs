using System.Collections.Generic;

namespace ML.Analyzer;

public static class AdamLayerGenerator
{
    private const string WeightType = "float";

    public static void GenerateAdam(SourceProductionContext context, Compilation compilation, LayerData data, AttributeData adamConfig)
    {
        var sb = new StringBuilder();

        var (layer, input, output, snapshot, weights) = data;

        if (adamConfig.NamedArguments.FirstOrDefault(p => p is { Key: "OutputGradientType", Value.Kind: TypedConstantKind.Type }) is { Key: not null } pair)
        {
            output = compilation.GetTypeByMetadataName(pair.Value.Value!.ToString())!;
        }

        sb.AppendLine($$"""
        using Ametrin.Guards;

        namespace {{layer.ContainingNamespace}};

        partial class {{layer.Name}}
        {
            public sealed class Adam : MachineLearning.Training.Optimization.ILayerOptimizer
            {
                public {{layer.Name}} Layer { get; }
                public MachineLearning.Training.Optimization.Adam.AdamOptimizer Optimizer { get; }

        """);

        #region Moments

        foreach (var weight in weights)
        {
            sb.AppendLine($"\t\tpublic {weight.Type} FirstMoment{weight.Name} {{ get; }}");
            sb.AppendLine($"\t\tpublic {weight.Type} SecondMoment{weight.Name} {{ get; }}");
            sb.AppendLine();
        }
        #endregion

        sb.AppendLine($$"""
                public Adam(MachineLearning.Training.Optimization.Adam.AdamOptimizer optimizer, {{layer.Name}} layer)
                {
                    this.Optimizer = optimizer;
                    this.Layer = layer;
        """);

        foreach (var weight in weights)
        {
            sb.AppendLine($"\t\t\tthis.FirstMoment{weight.Name} = {weight.Type}.OfSize(layer.{weight.Name});");
            sb.AppendLine($"\t\t\tthis.SecondMoment{weight.Name} = {weight.Type}.OfSize(layer.{weight.Name});");
        }

        sb.AppendLine($$"""
                }

                [System.Runtime.CompilerServices.ModuleInitializer]
                internal static void Register()
                {
                    MachineLearning.Training.Optimization.Adam.AdamOptimizer.Registry.Register<{{layer.Name}}>((op, l) => new Adam(op, l));
                }
        """);

        #region Update
        sb.AppendLine($$"""
                public void Update(Vector costGradient, MachineLearning.Model.Layer.Snapshot.ILayerSnapshot snapshot, MachineLearning.Model.Layer.Snapshot.IGradients gradients)
                {
                    var g = Guard.Is<{{layer.Name}}.Gradients>(gradients);
                    var s = Guard.Is<{{snapshot}}>(snapshot);
                    Layer.Backward({{(IsVector(output) ? "costGradient" : $"{output}.OfSize(s.Output, costGradient)")}}, s, g);

        """);

        foreach (var weight in weights)
        {
            sb.AppendLine($"\t\t\tNumericsDebug.AssertValidNumbers(g.{weight.Name});");
        }

        sb.AppendLine($$"""
                }
        """);

        #endregion

        #region Apply
        sb.Append($$"""
                public void Apply(MachineLearning.Model.Layer.Snapshot.IGradients gradients)
                {
                    if(gradients is not {{layer.Name}}.Gradients gradient)
                    {
                        throw new Exception();
                    }
        """);

        foreach (var weight in weights)
        {
            sb.AppendLine();
            sb.AppendLine($"\t\t\t(FirstMoment{weight.Name}, gradient.{weight.Name}).MapToFirst(FirstMomentEstimate);");
            sb.AppendLine($"\t\t\t(SecondMoment{weight.Name}, gradient.{weight.Name}).MapToFirst(SecondMomentEstimate);");
            sb.AppendLine($"\t\t\tLayer.{weight.Name}.SubtractToSelf((FirstMoment{weight.Name}, SecondMoment{weight.Name}).Map(WeightReduction));");
        }

        sb.AppendLine($$"""
                }

                private {{WeightType}} WeightReduction({{WeightType}} firstMoment, {{WeightType}} secondMoment)
                {
                    var mHat = firstMoment / (1 - Weight.Pow(Optimizer.FirstDecayRate, Optimizer.Iteration));
                    var vHat = secondMoment / (1 - Weight.Pow(Optimizer.SecondDecayRate, Optimizer.Iteration));
                    return Optimizer.LearningRate * mHat / (Weight.Sqrt(vHat) + Optimizer.Epsilon);
                }

                private {{WeightType}} FirstMomentEstimate({{WeightType}} lastMoment, {{WeightType}} gradient) => Optimizer.FirstDecayRate * lastMoment + (1 - Optimizer.FirstDecayRate) * gradient;

                private {{WeightType}} SecondMomentEstimate({{WeightType}} lastMoment, {{WeightType}} gradient) => Optimizer.SecondDecayRate * lastMoment + (1 - Optimizer.SecondDecayRate) * gradient * gradient;

        """);
        #endregion

        #region Reset

        sb.AppendLine($$"""
                public void FullReset()
                {
                    {{string.Join("\n\t\t\t", weights.Select(w => $"FirstMoment{w.Name}.ResetZero();"))}}
                    
                    {{string.Join("\n\t\t\t", weights.Select(w => $"SecondMoment{w.Name}.ResetZero();"))}}
                }
        """);

        #endregion

        sb.AppendLine($$"""
            }
        }    
        """);

        context.AddSource($"{layer.Name}.Adam.g.cs", sb.ToString());
    }
}

public sealed class LayerData(INamedTypeSymbol type, ITypeSymbol inputType, ITypeSymbol outputType, ITypeSymbol snapshotType, IEnumerable<IPropertySymbol> weights)
{
    public INamedTypeSymbol Type { get; } = type;
    public ITypeSymbol InputType { get; } = inputType;
    public ITypeSymbol OutputType { get; } = outputType;
    public ITypeSymbol SnapshotType { get; } = snapshotType;
    public IEnumerable<IPropertySymbol> Weights { get; } = weights;

    public void Deconstruct(out INamedTypeSymbol type, out ITypeSymbol inputType, out ITypeSymbol outputType, out ITypeSymbol snapshotType, out IEnumerable<IPropertySymbol> weights)
    {
        type = Type;
        inputType = InputType;
        outputType = OutputType;
        snapshotType = SnapshotType;
        weights = Weights;
    }
}