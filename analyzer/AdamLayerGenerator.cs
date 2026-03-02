using System.Collections.Generic;
using ML.Analyzer.LayerFile;

namespace ML.Analyzer;

internal static class AdamLayerGenerator
{
    private const string WeightType = "float";

    public static void GenerateAdam(SourceProductionContext context, LayerData data)
    {
        var sb = new StringBuilder();

        var (name, @namespace, _, output, snapshot, weights, modules) = data;

        sb.AppendLine($$"""
        using Ametrin.Guards;
        using Ametrin.Numerics;
        """);

        if (!string.IsNullOrEmpty(@namespace))
        {
            sb.AppendLine($$"""namespace {{@namespace}};""");
        }

        sb.AppendLine($$"""
        partial class {{name}}
        {
            public sealed class Adam : MachineLearning.Training.Optimization.ILayerOptimizer
            {
                public {{name}} Layer { get; }
                public MachineLearning.Training.Optimization.Adam.AdamOptimizer Optimizer { get; }

        """);

        #region Moments

        foreach (var weight in weights)
        {
            sb.AppendLine($"        public {weight.Type} FirstMoment{weight.Name} {{ get; }}");
            sb.AppendLine($"        public {weight.Type} SecondMoment{weight.Name} {{ get; }}");
            sb.AppendLine();
        }

        foreach (var module in modules)
        {
            sb.AppendLine($"        public {module.Type}.Adam {module.Name}Adam {{ get; }}");
        }
        #endregion

        sb.AppendLine($$"""
                public Adam(MachineLearning.Training.Optimization.Adam.AdamOptimizer optimizer, {{name}} layer)
                {
                    this.Optimizer = optimizer;
                    this.Layer = layer;
        """);

        foreach (var weight in weights)
        {
            sb.AppendLine($"            this.FirstMoment{weight.Name} = {weight.Type}.OfSize(layer.{weight.Name});");
            sb.AppendLine($"            this.SecondMoment{weight.Name} = {weight.Type}.OfSize(layer.{weight.Name});");
        }

        foreach (var module in modules)
        {
            sb.AppendLine($"            this.{module.Name}Adam = new(optimizer, {module.Access(LayerFile.Location.Gradients)});");
        }

        sb.AppendLine($$"""
                }

                [System.Runtime.CompilerServices.ModuleInitializer]
                internal static void Register()
                {
                    MachineLearning.Training.Optimization.Adam.AdamOptimizer.Registry.Register<{{name}}>((op, l) => new Adam(op, l));
                }
        """);

        #region Update
        sb.AppendLine($$"""
                public Vector Update(Vector costGradient, MachineLearning.Model.Layer.Snapshot.ILayerSnapshot snapshot, MachineLearning.Model.Layer.Snapshot.IGradients gradients)
                {
                    var g = Guard.Is<{{name}}.Gradients>(gradients);
                    var s = Guard.Is<{{snapshot}}>(snapshot);
                    var result = Layer.Backward({{(output is NumberType.Vector ? "costGradient" : $"{output}.Of(costGradient.Count / s.Output.ColumnCount, s.Output.ColumnCount, costGradient)")}}, s, g){{(output is NumberType.Vector ? "" : ".Storage")}};

        """);

        foreach (var weight in weights)
        {
            sb.AppendLine($"\t\t\tNumericsDebug.AssertValidNumbers(g.{weight.GetGradientName()});");
        }

        sb.AppendLine($$"""
                    return result;
                }
        """);

        #endregion

        #region Apply
        sb.AppendLine($$"""
                public void Apply(MachineLearning.Model.Layer.Snapshot.IGradients gradients)
                {
                    if(gradients is not {{name}}.Gradients gradient)
                    {
                        throw new Exception();
                    }
        """);

        if (weights.Any())
        {
            sb.Append($$"""
                    {{WeightType}} max;
        """);
        }

        foreach (var weight in weights)
        {
            sb.AppendLine($$"""

                    max = Weight.Abs(gradient.{{weight.GetGradientName()}}.MaxMagnitude());
                    if(max > 100_000)
                    {
                        gradient.{{weight.GetGradientName()}}.DivideToSelf(max/100_000);
                    }
                    (FirstMoment{{weight.Name}}, gradient.{{weight.GetGradientName()}}).MapToFirst(Optimizer.FirstMomentEstimate);
                    NumericsDebug.AssertValidNumbers(FirstMoment{{weight.Name}});
                    (SecondMoment{{weight.Name}}, gradient.{{weight.GetGradientName()}}).MapToFirst(Optimizer.SecondMomentEstimate);
                    NumericsDebug.AssertValidNumbers(SecondMoment{{weight.Name}});
                    Layer.{{weight.Name}}.SubtractToSelf((FirstMoment{{weight.Name}}, SecondMoment{{weight.Name}}).Map(Optimizer.WeightReduction));
        """);
        }

        sb.AppendLine();

        foreach (var module in modules)
        {
            sb.AppendLine($$"""
                    {{module.Name}}Adam.Apply(gradient.{{module.Name}});
        """);
        }
        #endregion

        #region Reset

        sb.AppendLine($$"""
                }

                public void FullReset()
                {
                    {{string.Join("\n            ", weights.Select(w => $"FirstMoment{w.Name}.ResetZero();"))}}
                    
                    {{string.Join("\n            ", weights.Select(w => $"SecondMoment{w.Name}.ResetZero();"))}}

                    {{string.Join("\n            ", modules.Select(m => $"{m.Name}Adam.FullReset();"))}}
                }
        """);

        #endregion

        sb.AppendLine($$"""
            }
        }    
        """);

        context.AddSource($"{name}.Adam.g.cs", sb.ToString());
    }
}

internal sealed record LayerData(string Name, string? Namespace, string InputType, NumberType OutputType, string SnapshotType, IEnumerable<DirectWeights> Weights, IEnumerable<Module> Modules);