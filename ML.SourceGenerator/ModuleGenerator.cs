using System.Text;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace ML.SourceGenerator;

[Generator]
public sealed class ModuleGenerator : IIncrementalGenerator
{
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var layers = context.SyntaxProvider.CreateSyntaxProvider(
            static (node, _) => node is ClassDeclarationSyntax { AttributeLists.Count: > 0 },
            static (ctx, token) => ctx.SemanticModel.GetDeclaredSymbol(ctx.Node, token) as INamedTypeSymbol
        ).Where(symbol => symbol!.HasAttribute(IsGeneratedModuleAttribute) && ImplementsIModule(symbol!));


        context.RegisterSourceOutput(layers.Combine(context.CompilationProvider), GenerateModule);
    }

    private static void GenerateModule(SourceProductionContext context, (INamedTypeSymbol?, Compilation) pair)
    {
        var (module, compilation) = pair;
        Debug.Assert(module is not null);
        var moduleDefinitionString = module.ToDisplayString(SymbolDisplayFormat.MinimallyQualifiedFormat);

        var attribute = module.TryGetAttribute(IsGeneratedModuleAttribute);
        Debug.Assert(attribute is not null);

        var moduleInfo = GetIModule(module);
        Debug.Assert(moduleInfo is not null);

        var modules = module.GetProperties().Where(static p => p.HasAttribute(IsSubModuleAttribute)).Select(SubModuleInfo.FromProperty).OfType<SubModuleInfo>().ToImmutableArray();
        var weights = module.GetProperties().Where(static p => p.HasAttribute(IsWeightAttribute)).ToImmutableArray();

        var generateDataClasses = moduleInfo.SnapshotType is null;
        if (generateDataClasses)
        {
            moduleInfo = moduleInfo with
            {
                SnapshotTypeString = $$"""{{moduleDefinitionString}}.Snapshot""",
                GradientsTypeString = $$"""{{moduleDefinitionString}}.Gradients""",
            };
        }

        var sb = new StringBuilder();

        sb.AppendLine($$"""#nullable enable""");
        sb.AppendLine();

        if (!module.ContainingNamespace.IsGlobalNamespace)
        {
            sb.AppendLine($$"""namespace {{module.ContainingNamespace}};""");
            sb.AppendLine();
        }


        sb.Append($$"""
        partial class {{moduleDefinitionString}}
        """);
        if (generateDataClasses)
        {
            sb.AppendLine(moduleInfo.RootModule.Name switch
            {
                ModuleName => $$""" : {{ModuleName}}<{{moduleInfo.ArchType}}, {{moduleInfo.SnapshotTypeString}}, {{moduleInfo.GradientsTypeString}}>""",
                HiddenModuleName => $$""" : {{HiddenModuleName}}<{{moduleInfo.ArchType}}, {{moduleInfo.SnapshotTypeString}}, {{moduleInfo.GradientsTypeString}}>""",
                _ => throw new NotImplementedException($"cannot impl {moduleInfo.RootModule.Name}"),
            });
        }
        else
        {
            sb.AppendLine();
        }

        var parameterCount = string.Join(" + ", [.. modules.Select(m => $"{m.Name}.ParameterCount"), .. weights.Select(w => $"(ulong){w.Name}.FlatCount")]);
        if (string.IsNullOrEmpty(parameterCount))
        {
            parameterCount = "0";
        }
        sb.AppendLine($$"""
        {
            public ulong ParameterCount => {{parameterCount}};


            public {{moduleInfo.SnapshotTypeString}} CreateSnapshot() => new(this);
            public {{moduleInfo.GradientsTypeString}} CreateGradients() => new(this);
        
        """);

        if (generateDataClasses)
        {
            GenerateSnapshot(sb, moduleDefinitionString, modules, weights);
            sb.AppendLine();
            GenerateGradients(sb, moduleDefinitionString, modules, weights);
        }


        sb.AppendLine($$"""
        }
        """);

        context.AddSource($"{module.Name}.g.cs", sb.ToString());
    }

    private static void GenerateSnapshot(StringBuilder sb, string moduleDefinitionString, IEnumerable<SubModuleInfo> modules, IEnumerable<IPropertySymbol> weights)
    {
        sb.AppendLine($$"""
            public sealed partial class Snapshot({{moduleDefinitionString}} module) : IModuleSnapshot
            {
        """);

        foreach (var sub in modules)
        {
            sb.AppendLine($$"""
                public {{sub.Module.SnapshotTypeString}} {{sub.Name}} { get; } = module.{{sub.Name}}.CreateSnapshot();
        """);
        }

        sb.AppendLine($$"""
            }
        """);
    }

    private static void GenerateGradients(StringBuilder sb, string moduleDefinitionString, IEnumerable<SubModuleInfo> modules, IEnumerable<IPropertySymbol> weights)
    {
        sb.AppendLine($$"""
            public sealed partial class Gradients({{moduleDefinitionString}} module) : IModuleGradients<Gradients>
            {
        """);

        foreach (var sub in modules)
        {
            sb.AppendLine($$"""
                public {{sub.Module.GradientsTypeString}} {{sub.Name}} { get; } = module.{{sub.Name}}.CreateGradients();
        """);
        }

        foreach (var weight in weights)
        {
            sb.AppendLine($$"""
                public {{weight.Type}} {{weight.Name}} { get; } = {{weight.Type}}.OfSize(module.{{weight.Name}});
        """);
        }

        sb.AppendLine($$"""
        
                public void Add(Gradients other)
                {
        """);

        foreach (var module in modules)
        {
            sb.AppendLine($$"""
                    {{module.Name}}.Add(other.{{module.Name}});
        """);
        }

        foreach (var weight in weights)
        {
            sb.AppendLine($$"""
                    {{weight.Name}}.AddToSelf(other.{{weight.Name}});
        """);
        }

        sb.AppendLine($$"""
                }

                public void Reset()
                {
        """);

        foreach (var module in modules)
        {
            sb.AppendLine($$"""
                    {{module.Name}}.Reset();
        """);
        }

        foreach (var weight in weights)
        {
            sb.AppendLine($$"""
                    {{weight.Name}}.ResetZero();
        """);
        }

        sb.AppendLine($$"""
                }
            }
        """);
    }
}
