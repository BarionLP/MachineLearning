using System.Collections.Immutable;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;

namespace ML.Analyzer;

[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class AdamLayerOptimizerAnalyzer : DiagnosticAnalyzer
{
    private static readonly DiagnosticDescriptor WeightsMustBeTensor = new(
        "ML001", "Non-Tensor used as Weights", "WeightsAttribute can only be used on Vectors and Matrices", "Usage", DiagnosticSeverity.Warning, isEnabledByDefault: true
    );

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [WeightsMustBeTensor];

    public override void Initialize(AnalysisContext context)
    {
        // context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.Analyze | GeneratedCodeAnalysisFlags.ReportDiagnostics);
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterSymbolAction(AnalyzePropertySymbol, SymbolKind.Property);
    }

    private static void AnalyzePropertySymbol(SymbolAnalysisContext context)
    {
        var propertySymbol = (IPropertySymbol)context.Symbol;
        if (propertySymbol.GetAttributes().Any(a => IsWeightAttribute(a.AttributeClass!)))
        {
            // if (!(IsVector(propertySymbol.Type!) || IsMatrix(propertySymbol.Type!)))
            // {
            // }
        }
        var diagnostic = Diagnostic.Create(WeightsMustBeTensor, propertySymbol.Locations[0]);
        context.ReportDiagnostic(diagnostic);
    }

    private static bool IsWeightAttribute(ITypeSymbol symbol) => symbol.Name == "WeightsAttribute" && symbol.ContainingNamespace.Name == "MachineLearning.Model.Attributes";
    private static bool IsLayerAttribute(ITypeSymbol symbol) => symbol.Name == "LayerAttribute" && symbol.ContainingNamespace.Name == "MachineLearning.Model.Attributes";
    private static bool IsVector(ITypeSymbol symbol) => symbol.Name == "Vector" && symbol.ContainingNamespace.Name == "Ametrin.Numerics";
    private static bool IsMatrix(ITypeSymbol symbol) => symbol.Name == "Matrix" && symbol.ContainingNamespace.Name == "Ametrin.Numerics";
}

[Generator]
public sealed class AdamLayerOptimizerGenerator : IIncrementalGenerator
{
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var layers = context.SyntaxProvider.CreateSyntaxProvider(
            static (node, _) => node is ClassDeclarationSyntax { AttributeLists.Count: > 0 },
            static (ctx, token) => ctx.SemanticModel.GetDeclaredSymbol(ctx.Node, token) as INamedTypeSymbol
        ).Where(symbol => symbol!.GetAttributes().Any(a => IsLayerAttribute(a.AttributeClass!)));

        context.RegisterSourceOutput(layers, GenerateLayer);
    }

    private static void GenerateLayer(SourceProductionContext context, INamedTypeSymbol? layer)
    {
        if (layer is null) return;

        var weights = layer.GetMembers().OfType<IPropertySymbol>().Where(p => p.GetAttributes().Any(a => IsWeightAttribute(a.AttributeClass!)));

        foreach (var weight in weights)
        {
            // if (!(IsVector(weight.Type!) || IsVector(weight.Type!)))
            // {
            //     var diagnostic = Diagnostic.Create(WeightsMustBeTensor, weight.Locations[0]);
            //     context.ReportDiagnostic(diagnostic);
            // }
        }

        var sb = new StringBuilder();
    }



    private static bool IsWeightAttribute(ITypeSymbol symbol) => symbol.Name == "WeightsAttribute" && symbol.ContainingNamespace.Name == "MachineLearning.Model.Attributes";
    private static bool IsLayerAttribute(ITypeSymbol symbol) => symbol.Name == "LayerAttribute" && symbol.ContainingNamespace.Name == "MachineLearning.Model.Attributes";
    private static bool IsVector(ITypeSymbol symbol) => symbol.Name == "Vector" && symbol.ContainingNamespace.Name == "Ametrin.Numerics";
    private static bool IsMatrix(ITypeSymbol symbol) => symbol.Name == "Matrix" && symbol.ContainingNamespace.Name == "Ametrin.Numerics";
}