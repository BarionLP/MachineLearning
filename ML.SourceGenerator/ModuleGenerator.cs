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

        var moduleInterface = GetIModule(module);
        Debug.Assert(moduleInterface is not null);

        var modules = module.GetProperties().Where(static p => p.HasAttribute(IsSubModuleAttribute)).ToImmutableArray();
        var weights = module.GetProperties().Where(static p => p.HasAttribute(IsWeightAttribute)).ToImmutableArray();

        var tarch = moduleInterface.TypeArguments[0];
        var generateSnapshot = moduleInterface.TypeArguments.Length == 1;

        var sb = new StringBuilder();

        sb.AppendLine($$"""#nullable enable""");
        sb.AppendLine();

        if (!module.ContainingNamespace.IsGlobalNamespace)
        {
            sb.AppendLine($$"""namespace {{module.ContainingNamespace}};""");
            sb.AppendLine();
        }

        var parameterCountCalc = string.Join(" + ", [.. modules.Select(m => $"{m.Name}.ParameterCount"), .. weights.Select(w => $"(ulong){w.Name}.FlatCount")]);

        sb.Append($$"""
        partial class {{moduleDefinitionString}}
        """);
        if (generateSnapshot)
        {
            sb.AppendLine($$""" : IModule<{{tarch}}, {{moduleDefinitionString}}.Snapshot, {{moduleDefinitionString}}.Gradients>""");
        }
        else
        {
            sb.AppendLine();
        }



        sb.AppendLine($$"""
        {
            public ulong ParameterCount => {{parameterCountCalc}};


            public Snapshot CreateSnapshot() => new(this);
            public Gradients CreateGradients() => new(this);        
        
        """);

        GenerateSnapshot(sb, moduleDefinitionString, modules, weights);
        sb.AppendLine();
        GenerateGradients(sb, moduleDefinitionString, modules, weights);

        sb.AppendLine($$"""
        }
        """);

        context.AddSource($"{module.Name}.g.cs", sb.ToString());
    }

    private static void GenerateSnapshot(StringBuilder sb, string moduleDefinitionString, IEnumerable<IPropertySymbol> modules, IEnumerable<IPropertySymbol> weights)
    {
        sb.AppendLine($$"""
            public sealed partial class Snapshot({{moduleDefinitionString}} module) : IModuleSnapshot
            {
        """);

        foreach (var module in modules)
        {
            var moduleSymbol = GetIModule(module.Type)!;
            var snapshotTypeName = moduleSymbol.TypeArguments.Length is 1 ? "IModuleSnapshot" : moduleSymbol.TypeArguments[1].ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            sb.AppendLine($$"""
                public {{snapshotTypeName}} {{module.Name}} { get; } = module.{{module.Name}}.CreateSnapshot();
        """);
        }

        sb.AppendLine($$"""
            }
        """);
    }

    private static void GenerateGradients(StringBuilder sb, string moduleDefinitionString, IEnumerable<IPropertySymbol> modules, IEnumerable<IPropertySymbol> weights)
    {
        sb.AppendLine($$"""
            public sealed partial class Gradients({{moduleDefinitionString}} module) : IModuleGradients<Gradients>
            {
        """);

        foreach (var module in modules)
        {
            var moduleSymbol = GetIModule(module.Type)!;
            var gradientTypeName = moduleSymbol.TypeArguments.Length is 1 ? "IModuleGradients" : moduleSymbol.TypeArguments[2].ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            sb.AppendLine($$"""
                public {{gradientTypeName}} {{module.Name}} { get; } = module.{{module.Name}}.CreateGradients();
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
