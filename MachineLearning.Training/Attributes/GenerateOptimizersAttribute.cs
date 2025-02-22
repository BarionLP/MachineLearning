namespace MachineLearning.Training.Attributes;

[AttributeUsage(AttributeTargets.Class)]
public sealed class GenerateOptimizersAttribute : Attribute
{
    public Type? OutputGradientType { get; init; }
}