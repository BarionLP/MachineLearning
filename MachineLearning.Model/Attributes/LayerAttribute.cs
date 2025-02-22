namespace MachineLearning.Model.Attributes;

[AttributeUsage(AttributeTargets.Class)]
public sealed class LayerAttribute : Attribute;

[AttributeUsage(AttributeTargets.Property)]
public sealed class WeightsAttribute : Attribute;