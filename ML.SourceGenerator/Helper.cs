using System.Text;

namespace ML.SourceGenerator;

internal static class Helper
{
    public const string CoreAssemblyName = "ML.Core";
    public const string NumericsAssemblyName = "Ametrin.Numerics";
    public const string ModuleName = "IModule";
    public const string InputModuleName = "IInputModule";
    public const string HiddenModuleName = "IHiddenModule";
    public const string OutputModuleName = "IOutputModule";

    public static bool IsIModule(INamedTypeSymbol? symbol)
        => symbol is { Name: ModuleName or InputModuleName or HiddenModuleName or OutputModuleName, ContainingAssembly.Name: CoreAssemblyName, ContainingNamespace.Name: "Modules", IsGenericType: true };

    public static bool ImplementsIModule(ITypeSymbol symbol)
        => symbol.Interfaces.Any(static i => IsIModule(i) || ImplementsIModule(i)) || (symbol.BaseType is not null && ImplementsIModule(symbol.BaseType));

    public static INamedTypeSymbol? GetIModule(ITypeSymbol symbol)
    {
        if (symbol is INamedTypeSymbol self && IsIModule(self))
        {
            return self;
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

    public static bool IsEmptyModuleData(ITypeSymbol? symbol) => symbol is { Name: "EmptyModuleData", ContainingAssembly.Name: CoreAssemblyName, ContainingNamespace.Name: "Modules" };
    public static bool IsSubModuleAttribute(INamedTypeSymbol? symbol) => symbol is { Name: "SubModuleAttribute", ContainingAssembly.Name: CoreAssemblyName, ContainingNamespace.Name: "Attributes" };
    public static bool IsWeightAttribute(INamedTypeSymbol? symbol) => symbol is { Name: "WeightsAttribute", ContainingAssembly.Name: CoreAssemblyName, ContainingNamespace.Name: "Attributes" };
    public static bool IsPropertyAttribute(INamedTypeSymbol? symbol) => symbol is { Name: "PropertyAttribute", ContainingAssembly.Name: CoreAssemblyName, ContainingNamespace.Name: "Attributes" };
    public static bool IsGeneratedModuleAttribute(INamedTypeSymbol? symbol) => symbol is { Name: "GeneratedModuleAttribute", ContainingAssembly.Name: CoreAssemblyName, ContainingNamespace.Name: "Attributes" };
    // public static bool IsGenerateOptimizersAttribute(INamedTypeSymbol? symbol) => symbol is { Name: "GenerateOptimizersAttribute", ContainingAssembly.Name: CoreAssemblyName, ContainingNamespace.Name: "Attributes" };
    public static bool IsGeneratedAdamAttribute(INamedTypeSymbol? symbol) => symbol is { Name: "GeneratedAdamAttribute", ContainingAssembly.Name: CoreAssemblyName, ContainingNamespace.Name: "Attributes" };
    public static bool IsVector(ITypeSymbol symbol) => symbol is { Name: "Vector", ContainingAssembly.Name: NumericsAssemblyName };
    public static bool IsMatrix(ITypeSymbol symbol) => symbol is { Name: "Matrix", ContainingAssembly.Name: NumericsAssemblyName };
    public static bool IsTensor(ITypeSymbol symbol) => symbol is { Name: "Tensor", ContainingAssembly.Name: NumericsAssemblyName };
    public static bool IsTensorLike(ITypeSymbol symbol) => symbol is { Name: "Vector" or "Matrix" or "Tensor", ContainingAssembly.Name: NumericsAssemblyName };

    public static int BuildFileHeaderFor(INamedTypeSymbol type, StringBuilder sb)
    {
        sb.AppendLine($$"""#nullable enable""");
        sb.AppendLine();

        if (!type.ContainingNamespace.IsGlobalNamespace)
        {
            sb.AppendLine($$"""namespace {{type.ContainingNamespace}};""");
            sb.AppendLine();
        }
        sb.AppendLine("""// ---- auto generated ----""");
        sb.AppendLine();

        var containers = new Stack<INamedTypeSymbol>();
        for (var t = type.ContainingType; t is not null; t = t.ContainingType) containers.Push(t);
        foreach (var c in containers)
        {
            BuildTypeHeader(c, sb);
            sb.AppendLine("\n{");
        }

        return containers.Count;
    }
    public static void BuildTypeHeader(INamedTypeSymbol type, StringBuilder sb)
    {
        var kind = type switch
        {
            { IsRecord: true, TypeKind: TypeKind.Struct } => "record struct",
            { IsRecord: true, TypeKind: TypeKind.Class } => "record",
            { TypeKind: TypeKind.Struct } => "struct",
            _ => "class",
        };

        sb.Append(type.IsStatic ? "static " : "");
        sb.Append(type is { IsRefLikeType: true, TypeKind: TypeKind.Struct } ? "ref " : "");
        sb.Append("partial ");
        sb.Append(kind);
        sb.Append(' ');
        sb.Append(type.Name);
    }

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


public sealed record ModulePropertyInfo(IPropertySymbol Property, SubModuleInfo Module)
{
    public string Name => Property.Name;
    public static ModulePropertyInfo? FromProperty(IPropertySymbol property) => SubModuleInfo.Create((INamedTypeSymbol)property.Type, canGenerateDataClasses: false) is { } module ? new(property, module) : null;
}


public sealed record ModuleInfo(INamedTypeSymbol Type, ImmutableArray<ModulePropertyInfo> Modules, ImmutableArray<IPropertySymbol> Weights, INamedTypeSymbol RootModule, string ModuleDefinitionString, ITypeSymbol ArchType, bool GenerateDataClasses, ITypeSymbol? SnapshotType, ITypeSymbol? GradientsType)
    : SubModuleInfo(RootModule, ModuleDefinitionString, ArchType, GenerateDataClasses, SnapshotType, GradientsType);
public record SubModuleInfo(INamedTypeSymbol RootModule, string ModuleDefinitionString, ITypeSymbol ArchType, bool GenerateDataClasses, ITypeSymbol? SnapshotType, ITypeSymbol? GradientsType)
{
    public string SnapshotTypeString { get; init; } = SnapshotType is null ? "IModuleSnapshot" : SnapshotType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
    public string GradientsTypeString { get; init; } = GradientsType is null ? "IModuleGradients" : GradientsType.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);

    public static SubModuleInfo? Create(INamedTypeSymbol type, bool canGenerateDataClasses)
    {
        var module = GetIModule(type);
        if (module is null) return null;

        var moduleDefinitionString = type.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat);

        var archType = module.Name switch
        {
            InputModuleName => module.TypeArguments[1],
            _ => module.TypeArguments[0],
        };

        var (snapshot, gradients) = module.TypeArguments.Length > 2 ? module.Name switch
        {
            InputModuleName or OutputModuleName => (module.TypeArguments[2], module.TypeArguments[3]),
            _ => (module.TypeArguments[1], module.TypeArguments[2]),
        } : (null, null);

        var generateDataClasses = canGenerateDataClasses && snapshot is null;
        if (generateDataClasses)
        {
            return new(module, moduleDefinitionString, archType, generateDataClasses, snapshot, gradients)
            {
                SnapshotTypeString = $$"""{{moduleDefinitionString}}.Snapshot""",
                GradientsTypeString = $$"""{{moduleDefinitionString}}.Gradients""",
            };
        }

        return new(module, moduleDefinitionString, archType, generateDataClasses, snapshot, gradients);
    }

    public static ModuleInfo? CreateFull(INamedTypeSymbol type, bool canGenerateDataClasses)
    {

        var sub = Create(type, canGenerateDataClasses);

        if (sub is null) return null;

        var modules = type.GetProperties().Where(static p => p.HasAttribute(IsSubModuleAttribute)).Select(ModulePropertyInfo.FromProperty).OfType<ModulePropertyInfo>().ToImmutableArray();
        var weights = type.GetProperties().Where(static p => p.HasAttribute(IsWeightAttribute)).ToImmutableArray();

        return new(type, modules, weights, sub.RootModule, sub.ModuleDefinitionString, sub.ArchType, sub.GenerateDataClasses, sub.SnapshotType, sub.GradientsType)
        {
            SnapshotTypeString = sub.SnapshotTypeString,
            GradientsTypeString = sub.GradientsTypeString,
        };
    }
}