using System.Collections.Generic;

namespace ML.Analyzer;

public static class AdamLayerGenerator
{
    public static void GenerateAdam(SourceProductionContext context, Compilation compilation, LayerData data, AttributeData adamConfig)
    {
        var sb = new StringBuilder();

        var (layer, input, output, weights) = data;

        if (adamConfig.NamedArguments.FirstOrDefault(p => p is { Key: "OutputGradientType", Value.Kind: TypedConstantKind.Type }) is { Key: not null } pair)
        {
            output = compilation.GetTypeByMetadataName(pair.Value.Value!.ToString())!;
        }

        sb.AppendLine($$"""
        namespace {{layer.ContainingNamespace}};

        partial class {{layer.Name}}
        {
            public sealed class Adam : MachineLearning.Training.Optimization.ILayerOptimizer<{{layer.Name}}, {{layer.Name}}.Snapshot>
            {
                public {{layer.Name}} Layer { get; }
                public MachineLearning.Training.Optimization.Adam.AdamOptimizer Optimizer { get; }

        """);

        #region Gradients

        foreach (var weight in weights)
        {
            sb.AppendLine($"\t\tpublic {weight.Type} Gradient{weight.Name} {{ get; }}");
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
            sb.AppendLine($"\t\t\tthis.Gradient{weight.Name} = {weight.Type}.OfSize(layer.{weight.Name});");
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
                private readonly Lock _lock = new();
                public void Update(Vector costGradient, {{layer.Name}}.Snapshot snapshot)
                {
                    Layer.Backward({{(IsVector(output) ? "costGradient" : $"{output}.OfSize(snapshot.Output, costGradient)")}}, snapshot);

        """);

        foreach (var weight in weights)
        {
            sb.AppendLine($"\t\t\tNumericsDebug.AssertValidNumbers(snapshot.Gradient{weight.Name});");
        }

        sb.AppendLine($$"""
                    lock (_lock)
                    {        
        """);

        foreach (var weight in weights)
        {
            sb.AppendLine($"\t\t\t\tGradient{weight.Name}.AddToSelf(snapshot.Gradient{weight.Name});");
        }

        sb.AppendLine($$"""
                    }
                }
        """);

        #endregion

        #region Apply
        sb.Append($$"""
                public void Apply(int _)
                {
        """);

        foreach (var weight in weights)
        {
            sb.AppendLine();
            sb.AppendLine($"\t\t\t(FirstMoment{weight.Name}, Gradient{weight.Name}).MapToFirst(FirstMomentEstimate);");
            sb.AppendLine($"\t\t\t(SecondMoment{weight.Name}, Gradient{weight.Name}).MapToFirst(SecondMomentEstimate);");
            sb.AppendLine($"\t\t\tLayer.{weight.Name}.SubtractToSelf((FirstMoment{weight.Name}, SecondMoment{weight.Name}).Map(WeightReduction));");
        }

        sb.AppendLine($$"""
                }

                private float WeightReduction(float firstMoment, float secondMoment)
                {
                    var mHat = firstMoment / (1 - Weight.Pow(Optimizer.FirstDecayRate, Optimizer.Iteration));
                    var vHat = secondMoment / (1 - Weight.Pow(Optimizer.SecondDecayRate, Optimizer.Iteration));
                    return Optimizer.LearningRate * mHat / (Weight.Sqrt(vHat) + Optimizer.Epsilon);
                }

                private float FirstMomentEstimate(float lastMoment, float gradient) => Optimizer.FirstDecayRate * lastMoment + (1 - Optimizer.FirstDecayRate) * gradient;

                private float SecondMomentEstimate(float lastMoment, float gradient) => Optimizer.SecondDecayRate * lastMoment + (1 - Optimizer.SecondDecayRate) * gradient * gradient;

        """);
        #endregion

        #region Reset

        sb.AppendLine($$"""
                public void GradientCostReset()
                {
                    {{string.Join("\n\t\t\t", weights.Select(w => $"Gradient{w.Name}.ResetZero();"))}}
                }


                public void FullReset()
                {
                    GradientCostReset();

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

public sealed class LayerData(INamedTypeSymbol type, ITypeSymbol inputType, ITypeSymbol outputType, IEnumerable<IPropertySymbol> weights)
{
    public INamedTypeSymbol Type { get; } = type;
    public ITypeSymbol InputType { get; } = inputType;
    public ITypeSymbol OutputType { get; } = outputType;
    public IEnumerable<IPropertySymbol> Weights { get; } = weights;

    public void Deconstruct(out INamedTypeSymbol type, out ITypeSymbol inputType, out ITypeSymbol outputType, out IEnumerable<IPropertySymbol> weights)
    {
        type = Type;
        inputType = InputType;
        outputType = OutputType;
        weights = Weights;
    }
}