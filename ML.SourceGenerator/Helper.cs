namespace ML.SourceGenerator;

internal static class Helper
{
    public const string CoreAssemblyName = "ML.Core";
    public const string NumericsAssemblyName = "Ametrin.Numerics";

    public static bool IsIModule(INamedTypeSymbol? symbol)
        => symbol is { Name: "IModule", ContainingAssembly.Name: CoreAssemblyName, ContainingNamespace.Name: "Modules", IsGenericType: true };

    public static bool ImplementsIModule(ITypeSymbol symbol)
        => symbol.Interfaces.Any(static i => IsIModule(i) || ImplementsIModule(i)) || (symbol.BaseType is not null && ImplementsIModule(symbol.BaseType));

    public static INamedTypeSymbol? GetIModule(ITypeSymbol symbol)
    {
        if (symbol is INamedTypeSymbol ts && IsIModule(ts))
        {
            return ts;
        }
        else if (symbol.Interfaces.Where(IsIModule).OrderByDescending(i => i.TypeArguments.Length).FirstOrDefault() is INamedTypeSymbol inter)
        {
            return inter;
        }
        else if (symbol.Interfaces.FirstOrDefault(ImplementsIModule) is INamedTypeSymbol inter2)
        {
            return GetIModule(inter2);
        }
        else
        {
            return symbol.BaseType is null ? null : GetIModule(symbol.BaseType);
        }

    }

    public static bool IsSubModuleAttribute(INamedTypeSymbol? symbol) => symbol is { Name: "SubModuleAttribute", ContainingAssembly.Name: CoreAssemblyName, ContainingNamespace.Name: "Attributes" };
    public static bool IsWeightAttribute(INamedTypeSymbol? symbol) => symbol is { Name: "WeightsAttribute", ContainingAssembly.Name: CoreAssemblyName, ContainingNamespace.Name: "Attributes" };
    // public static bool IsParameterAttribute(INamedTypeSymbol? symbol) => symbol is { Name: "ParameterAttribute", ContainingAssembly.Name: CoreAssemblyName, ContainingNamespace.Name: "Attributes" };
    public static bool IsGeneratedModuleAttribute(INamedTypeSymbol? symbol) => symbol is { Name: "GeneratedModuleAttribute", ContainingAssembly.Name: CoreAssemblyName, ContainingNamespace.Name: "Attributes" };
    public static bool IsGenerateOptimizersAttribute(INamedTypeSymbol? symbol) => symbol is { Name: "GenerateOptimizersAttribute", ContainingAssembly.Name: CoreAssemblyName, ContainingNamespace.Name: "Attributes" };
    public static bool IsLayerSerializerAttribute(INamedTypeSymbol? symbol) => symbol is { Name: "LayerSerializerAttribute", ContainingAssembly.Name: CoreAssemblyName };
    public static bool IsVector(ITypeSymbol symbol) => symbol is { Name: "Vector", ContainingAssembly.Name: NumericsAssemblyName };
    public static bool IsMatrix(ITypeSymbol symbol) => symbol is { Name: "Matrix", ContainingAssembly.Name: NumericsAssemblyName };
    public static bool IsTensor(ITypeSymbol symbol) => symbol is { Name: "Tensor", ContainingAssembly.Name: NumericsAssemblyName };
    public static bool IsTensorLike(ITypeSymbol symbol) => symbol is { Name: "Vector" or "Matrix" or "Tensor", ContainingAssembly.Name: NumericsAssemblyName };

    extension(ISymbol symbol)
    {
        public bool HasAttribute(Func<INamedTypeSymbol?, bool> condition) => symbol.GetAttributes().Select(static a => a.AttributeClass).Any(condition);
        public AttributeData? TryGetAttribute(Func<INamedTypeSymbol?, bool> condition) => symbol.GetAttributes().Where(a => condition(a.AttributeClass)).FirstOrDefault();

    }

    extension(ITypeSymbol symbol)
    {
        public IEnumerable<IPropertySymbol> GetProperties() => symbol.GetMembers().OfType<IPropertySymbol>();
    }
}
