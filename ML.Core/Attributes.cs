namespace ML.Core.Attributes;


[AttributeUsage(AttributeTargets.Property)]
public sealed class SubModuleAttribute : Attribute;

[AttributeUsage(AttributeTargets.Property)]
public sealed class WeightsAttribute : Attribute;

[AttributeUsage(AttributeTargets.Property)]
public sealed class PropertyAttribute : Attribute;

[AttributeUsage(AttributeTargets.Class)]
public sealed class GeneratedModuleAttribute : Attribute;