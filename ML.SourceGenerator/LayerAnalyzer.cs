using Microsoft.CodeAnalysis.Diagnostics;

namespace ML.SourceGenerator;

[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class LayerAnalyzer : DiagnosticAnalyzer
{
    private static readonly DiagnosticDescriptor WeightsMustBeTensors = new(
       "ML001", "Non-Tensor used as Weights", "WeightsAttribute can only be used on Tensors not {0}", "Usage", DiagnosticSeverity.Warning, isEnabledByDefault: true
    );
    private static readonly DiagnosticDescriptor InvalidGeneratedLayer = new(
        "ML002", "Invalid GeneratedLayerAttribute", "GeneratedLayerAttribute can only be used on instances of ILayer<TInput, TOutput, TSnapshot>", "Usage", DiagnosticSeverity.Warning, isEnabledByDefault: true
    );
    private static readonly DiagnosticDescriptor InvalidLayerSerializer = new(
        "ML003", "Invalid LayerSerializerAttribute", "LayerSerializerAttribute can only be used with GeneratedLayerAttribute", "Usage", DiagnosticSeverity.Error, isEnabledByDefault: true
    );

    private static readonly DiagnosticDescriptor SubModuleMustBeIModule = new(
       "ML004", "Non-IModule used as SubModule", "SubModuleAttribute can only be used IModule<TArch> not {0}", "Usage", DiagnosticSeverity.Error, isEnabledByDefault: true
    );


    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [WeightsMustBeTensors, InvalidGeneratedLayer, InvalidLayerSerializer, SubModuleMustBeIModule];

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.Analyze | GeneratedCodeAnalysisFlags.ReportDiagnostics);
        context.EnableConcurrentExecution();

        context.RegisterSymbolAction(AnalyzePropertySymbol, SymbolKind.Property);
        context.RegisterSymbolAction(AnalyzeClassSymbol, SymbolKind.NamedType);
    }

    private static void AnalyzePropertySymbol(SymbolAnalysisContext context)
    {
        var propertySymbol = (IPropertySymbol)context.Symbol;
        if (propertySymbol.HasAttribute(IsWeightAttribute))
        {
            if (!IsTensorLike(propertySymbol.Type))
            {
                context.ReportDiagnostic(Diagnostic.Create(WeightsMustBeTensors, propertySymbol.Locations[0], propertySymbol.Type));
            }
        }
        else if (propertySymbol.HasAttribute(IsSubModuleAttribute))
        {
            if (!ImplementsIModule(propertySymbol.Type))
            {
                context.ReportDiagnostic(Diagnostic.Create(SubModuleMustBeIModule, propertySymbol.Locations[0], propertySymbol.Type));
            }
            // var moduleSymbol = GetIModule(propertySymbol.Type)!;
            // var snapshotTypeName = moduleSymbol.TypeArguments.Length is 1 ? "IModuleSnapshot" : moduleSymbol.TypeArguments[1].ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat);
            // context.ReportDiagnostic(Diagnostic.Create(SubModuleMustBeIModule, propertySymbol.Locations[0], snapshotTypeName));
        }
    }

    private static void AnalyzeClassSymbol(SymbolAnalysisContext context)
    {
        var typeSymbol = (INamedTypeSymbol)context.Symbol;
        if (typeSymbol.HasAttribute(IsGeneratedModuleAttribute))
        {
            if (!ImplementsIModule(typeSymbol))
            {
                context.ReportDiagnostic(Diagnostic.Create(InvalidGeneratedLayer, typeSymbol.Locations[0]));
            }
        }
        else if (typeSymbol.HasAttribute(IsLayerSerializerAttribute))
        {
            context.ReportDiagnostic(Diagnostic.Create(InvalidLayerSerializer, typeSymbol.Locations[0]));
        }
    }
}
