namespace MachineLearning.Model.Attributes;

[AttributeUsage(AttributeTargets.Class)]
public sealed class GeneratedLayerAttribute : Attribute
{
    public Type? OutputGradientType { get; init; }
}
