using System.Collections.Generic;
using ML.Analyzer.LayerFile;

namespace ML.Analyzer;

internal static class AdamLayerGenerator
{
    private const string WeightType = "float";

    public static void GenerateAdam(SourceProductionContext context, LayerData data)
    {
        var sb = new StringBuilder();

        var (name, @namespace, _, output, snapshot, weights) = data;

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
            sb.AppendLine($"\t\tpublic {weight.Type} FirstMoment{weight.Name} {{ get; }}");
            sb.AppendLine($"\t\tpublic {weight.Type} SecondMoment{weight.Name} {{ get; }}");
            sb.AppendLine();
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
            sb.AppendLine($"\t\t\tthis.FirstMoment{weight.Name} = {weight.Type}.OfSize(layer.{weight.Name});");
            sb.AppendLine($"\t\t\tthis.SecondMoment{weight.Name} = {weight.Type}.OfSize(layer.{weight.Name});");
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
                public void Update(Vector costGradient, MachineLearning.Model.Layer.Snapshot.ILayerSnapshot snapshot, MachineLearning.Model.Layer.Snapshot.IGradients gradients)
                {
                    var g = Guard.Is<{{name}}.Gradients>(gradients);
                    var s = Guard.Is<{{snapshot}}>(snapshot);
                    Layer.Backward({{(output is NumberType.Vector ? "costGradient" : $"{output}.Of(costGradient.Count / s.Output.ColumnCount, s.Output.ColumnCount, costGradient)")}}, s, g);

        """);

        foreach (var weight in weights)
        {
            sb.AppendLine($"\t\t\tNumericsDebug.AssertValidNumbers(g.{weight.GetGradientName()});");
        }

        sb.AppendLine($$"""
                }
        """);

        #endregion

        #region Apply
        sb.Append($$"""
                public void Apply(MachineLearning.Model.Layer.Snapshot.IGradients gradients)
                {
                    if(gradients is not {{name}}.Gradients gradient)
                    {
                        throw new Exception();
                    }
                    {{WeightType}} max;
        """);

        foreach (var weight in weights)
        {
            sb.AppendLine($$"""

                    max = Weight.Abs(gradient.{{weight.GetGradientName()}}.MaxMagnitude());
                    if(max > 100_000)
                    {
                        gradient.{{weight.GetGradientName()}}.DivideToSelf(max/100_000);
                    }
                    (FirstMoment{{weight.Name}}, gradient.{{weight.GetGradientName()}}).MapToFirst(FirstMomentEstimate);
                    NumericsDebug.AssertValidNumbers(FirstMoment{{weight.Name}});
                    (SecondMoment{{weight.Name}}, gradient.{{weight.GetGradientName()}}).MapToFirst(SecondMomentEstimate);
                    NumericsDebug.AssertValidNumbers(SecondMoment{{weight.Name}});
                    Layer.{{weight.Name}}.SubtractToSelf((FirstMoment{{weight.Name}}, SecondMoment{{weight.Name}}).Map(WeightReduction));
        """);
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

        context.AddSource($"{name}.Adam.g.cs", sb.ToString());
    }
}

internal sealed record LayerData(string Name, string? Namespace, string InputType, NumberType OutputType, string SnapshotType, IEnumerable<DirectWeights> Weights);