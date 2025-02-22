using System;
using MachineLearning.Model.Layer;

namespace MachineLearning.Serialization;

#pragma warning disable CS9113 // Parameter is unread.
[AttributeUsage(AttributeTargets.Class)]
public sealed class LayerSerializerAttribute(string key, int version) : Attribute;
#pragma warning restore CS9113 // Parameter is unread.
