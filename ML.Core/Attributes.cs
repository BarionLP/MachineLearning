namespace ML.Core.Attributes;

#pragma warning disable CS9113 // Parameter is unread. only required by sourcegen
[AttributeUsage(AttributeTargets.Property)]
public sealed class SubModuleAttribute : Attribute;

[AttributeUsage(AttributeTargets.Property)]
public sealed class WeightsAttribute : Attribute;

[AttributeUsage(AttributeTargets.Property)]
public sealed class PropertyAttribute : Attribute;

[AttributeUsage(AttributeTargets.Class)]
public sealed class GeneratedModuleAttribute : Attribute;

[AttributeUsage(AttributeTargets.Class)]
public sealed class GeneratedAdamAttribute(Type module) : Attribute;
#pragma warning restore CS9113 // Parameter is unread.
