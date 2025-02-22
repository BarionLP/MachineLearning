namespace ML.Analyzer;

public static class Helper
{
    public static bool IsGenericILayer(INamedTypeSymbol symbol)
        => symbol is { Name: "ILayer", ContainingAssembly.Name: "MachineLearning.Model", ContainingNamespace.Name: "Layer", TypeArguments.Length: 3 };

    public static bool ImplementsGenericILayer(INamedTypeSymbol symbol) => symbol.Interfaces.Any(i => IsGenericILayer(i) || ImplementsGenericILayer(i)) || (symbol.BaseType is not null && IsGenericILayer(symbol.BaseType));

    public static INamedTypeSymbol? GetGenericILayer(INamedTypeSymbol symbol)
        => IsGenericILayer(symbol) ? symbol
        : symbol.Interfaces.FirstOrDefault(IsGenericILayer) is INamedTypeSymbol inter ? inter
        : symbol.Interfaces.FirstOrDefault(ImplementsGenericILayer) is INamedTypeSymbol inter2 ? GetGenericILayer(inter2)
        : symbol.BaseType is null ? null : GetGenericILayer(symbol.BaseType);

    public static bool IsWeightAttribute(ITypeSymbol symbol) => symbol is { Name: "WeightsAttribute", ContainingAssembly.Name: "MachineLearning.Model", ContainingNamespace.Name: "Attributes" };
    public static bool IsParameterAttribute(ITypeSymbol symbol) => symbol is { Name: "ParameterAttribute", ContainingAssembly.Name: "MachineLearning.Model", ContainingNamespace.Name: "Attributes" };
    public static bool IsGeneratedLayerAttribute(ITypeSymbol symbol) => symbol is { Name: "GeneratedLayerAttribute", ContainingAssembly.Name: "MachineLearning.Model", ContainingNamespace.Name: "Attributes" };
    public static bool IsGenerateOptimizersAttribute(ITypeSymbol symbol) => symbol is { Name: "GenerateOptimizersAttribute", ContainingAssembly.Name: "MachineLearning.Training", ContainingNamespace.Name: "Attributes" };
    public static bool IsLayerSerializerAttribute(ITypeSymbol symbol) => symbol is { Name: "LayerSerializerAttribute", ContainingAssembly.Name: "MachineLearning.Serialization" };
    public static bool IsVector(ITypeSymbol symbol) => symbol is { Name: "Vector", ContainingAssembly.Name: "Ametrin.Numerics" };
    public static bool IsMatrix(ITypeSymbol symbol) => symbol is { Name: "Matrix", ContainingAssembly.Name: "Ametrin.Numerics" };
}
