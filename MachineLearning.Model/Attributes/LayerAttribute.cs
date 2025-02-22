namespace MachineLearning.Model.Attributes;

[AttributeUsage(AttributeTargets.Class)]
public sealed class LayerAttribute<TArch> : Attribute;
