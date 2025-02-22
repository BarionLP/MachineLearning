using System.Collections.Immutable;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace ML.Analyzer;

[Generator]
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class AdamLayerOptimizerAnalyzer : DiagnosticAnalyzer, IIncrementalGenerator
{
    private static readonly DiagnosticDescriptor WeightsMustBeTensor = new(
        "ML001", "Non-Tensor used as Weights", "WeightsAttribute can only be used on Vectors and Matrices", "Usage", DiagnosticSeverity.Warning, isEnabledByDefault: true
    );

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [WeightsMustBeTensor];

    public override void Initialize(AnalysisContext context)
    {
        //context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.Analyze | GeneratedCodeAnalysisFlags.ReportDiagnostics);
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterSymbolAction(AnalyzePropertySymbol, SymbolKind.Property);
    }

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var layers = context.SyntaxProvider.CreateSyntaxProvider(
            static (node, _) => node is ClassDeclarationSyntax { AttributeLists.Count: > 0 },
            static (ctx, token) => ctx.SemanticModel.GetDeclaredSymbol(ctx.Node, token) as INamedTypeSymbol
        ).Where(symbol => symbol!.GetAttributes().Any(a => IsLayerAttribute(a.AttributeClass!)));

        context.RegisterSourceOutput(layers, GenerateLayer);
    }

    private static void AnalyzePropertySymbol(SymbolAnalysisContext context)
    {
        var propertySymbol = (IPropertySymbol)context.Symbol;
        if (propertySymbol.GetAttributes().Any(a => IsWeightAttribute(a.AttributeClass!)))
        {
            if (!(IsVector(propertySymbol.Type!) || IsMatrix(propertySymbol.Type!)))
            {
                var diagnostic = Diagnostic.Create(WeightsMustBeTensor, propertySymbol.Locations[0]);
                context.ReportDiagnostic(diagnostic);
            }
        }
    }

    private static void GenerateLayer(SourceProductionContext context, INamedTypeSymbol? layer)
    {
        if (layer is null) return;

        var layerAttribute = layer.GetAttributes().FirstOrDefault(a => IsLayerAttribute(a.AttributeClass!));

        if (layerAttribute is null) return;

        var arch = layerAttribute.AttributeClass!.TypeArguments[0];

        var weights = layer.GetMembers().OfType<IPropertySymbol>().Where(p => p.GetAttributes().Any(a => IsWeightAttribute(a.AttributeClass!)));

        var sb = new StringBuilder();
        sb.AppendLine($$"""
        using Ametrin.Numerics;
        using MachineLearning.Model.Layer.Snapshot;

        namespace {{layer.ContainingNamespace}};
        
        partial class {{layer.Name}} 
        {
        """);

        sb.AppendLine($$"""
            public ILayerSnapshot CreateSnapshot() => new Snapshot(this);


            public partial class Snapshot({{layer.Name}} layer) : ILayerSnapshot
            {
                public {{arch.Name}} Input { get; } = {{arch.Name}}.Create(T, E);
                public {{arch.Name}} Output { get; } = {{arch.Name}}.Create(T, E);
                public {{arch.Name}} GradientInput { get; } = {{arch.Name}}.Create(T, E);
        """);

        foreach (var weight in weights)
        {
            sb.AppendLine($"\t\tpublic {weight.Type} Gradient{weight.Name} {{ get; }} = {weight.Type}.OfSize(layer);");
        }

        sb.AppendLine("\t}");
        sb.AppendLine("}");

        context.AddSource($"{layer.Name}.g.cs", sb.ToString());
    }

    private static bool IsWeightAttribute(ITypeSymbol symbol) => symbol.Name == "WeightsAttribute" && symbol.ContainingAssembly.Name == "MachineLearning.Model" && symbol.ContainingNamespace.Name == "Attributes";
    private static bool IsLayerAttribute(ITypeSymbol symbol) => symbol.Name == "LayerAttribute" && symbol.ContainingAssembly.Name == "MachineLearning.Model" && symbol.ContainingNamespace.Name == "Attributes";
    private static bool IsVector(ITypeSymbol symbol) => symbol.Name == "Vector" && symbol.ContainingAssembly.Name == "Ametrin.Numerics";
    private static bool IsMatrix(ITypeSymbol symbol) => symbol.Name == "Matrix" && symbol.ContainingAssembly.Name == "Ametrin.Numerics";
}
