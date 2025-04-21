namespace ML.Analyzer;

[Generator]
[DiagnosticAnalyzer(LanguageNames.CSharp)]
public sealed class LayerAnalyzer : DiagnosticAnalyzer, IIncrementalGenerator
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
    private static readonly DiagnosticDescriptor BothParamWeightUsed = new(
        "ML004", "Invalid Layer configuration", "WeightsAttribute and ParameterAttribute cannot be used together", "Usage", DiagnosticSeverity.Error, isEnabledByDefault: true
    );

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = [WeightsMustBeTensors, InvalidGeneratedLayer, InvalidLayerSerializer, BothParamWeightUsed];

    public override void Initialize(AnalysisContext context)
    {
        context.ConfigureGeneratedCodeAnalysis(GeneratedCodeAnalysisFlags.None);
        context.EnableConcurrentExecution();
        context.RegisterSymbolAction(AnalyzePropertySymbol, SymbolKind.Property);
        context.RegisterSymbolAction(AnalyzeClassSymbol, SymbolKind.NamedType);
    }

    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var layers = context.SyntaxProvider.CreateSyntaxProvider(
            static (node, _) => node is ClassDeclarationSyntax { AttributeLists.Count: > 0 },
            static (ctx, token) => ctx.SemanticModel.GetDeclaredSymbol(ctx.Node, token) as INamedTypeSymbol
        ).Where(symbol => symbol!.GetAttributes().Any(a => IsGeneratedLayerAttribute(a.AttributeClass!)) && ImplementsGenericILayer(symbol));

        // Debugger.Launch();

        context.RegisterSourceOutput(layers.Combine(context.CompilationProvider), GenerateLayer);
    }

    private static void AnalyzePropertySymbol(SymbolAnalysisContext context)
    {
        var propertySymbol = (IPropertySymbol)context.Symbol;
        if (propertySymbol.GetAttributes().Any(a => IsWeightAttribute(a.AttributeClass!)))
        {
            if (!IsTensorlike(propertySymbol.Type!))
            {
                context.ReportDiagnostic(Diagnostic.Create(WeightsMustBeTensors, propertySymbol.Locations[0], propertySymbol.Type));
            }

            if (propertySymbol.GetAttributes().Any(a => IsParameterAttribute(a.AttributeClass!)))
            {
                context.ReportDiagnostic(Diagnostic.Create(BothParamWeightUsed, propertySymbol.Locations[0]));
            }
        }
    }

    private static void AnalyzeClassSymbol(SymbolAnalysisContext context)
    {
        var typeSymbol = (INamedTypeSymbol)context.Symbol;
        if (typeSymbol.GetAttributes().Any(a => IsGeneratedLayerAttribute(a.AttributeClass!)))
        {
            if (!ImplementsGenericILayer(typeSymbol))
            {
                context.ReportDiagnostic(Diagnostic.Create(InvalidGeneratedLayer, typeSymbol.Locations[0]));
            }
        }
        else if (typeSymbol.GetAttributes().Any(a => IsLayerSerializerAttribute(a.AttributeClass!)))
        {
            context.ReportDiagnostic(Diagnostic.Create(InvalidLayerSerializer, typeSymbol.Locations[0]));
        }
    }

    private static void GenerateLayer(SourceProductionContext context, (INamedTypeSymbol?, Compilation) pair)
    {
        var (layer, compilation) = pair;

        if (layer is null) return;

        var attribute = layer.GetAttributes().First(a => IsGeneratedLayerAttribute(a.AttributeClass!));

        var ilayer = GetGenericILayer(layer);

        if (ilayer is null) return;

        var tin = ilayer.TypeArguments[0];
        var tout = ilayer.TypeArguments[1];
        var tsnap = ilayer.TypeArguments[2];


        if (attribute.NamedArguments.FirstOrDefault(p => p is { Key: "OutputGradientType", Value.Kind: TypedConstantKind.Type }) is { Key: not null } p)
        {
            tout = compilation.GetTypeByMetadataName(p.Value.Value!.ToString())!;
        }

        var weights = layer.GetMembers().OfType<IPropertySymbol>().Where(p => p.GetAttributes().Any(a => IsWeightAttribute(a.AttributeClass!)));
        var parameter = layer.GetMembers().OfType<IPropertySymbol>().Where(p => p.GetAttributes().Any(a => IsParameterAttribute(a.AttributeClass!)));

        var sb = new StringBuilder();
        sb.AppendLine($$"""
        using Ametrin.Guards;
        using MachineLearning.Model.Layer.Snapshot;

        namespace {{layer.ContainingNamespace}};
        
        partial class {{layer.Name}} 
        {
            public {{layer.Name}}({{string.Join(", ", parameter.Concat(weights).Select(p => $"{p.Type} {p.Name.ToLower()}"))}})
            {
                {{string.Join("\n\t\t", parameter.Concat(weights).Select(p => $"this.{p.Name} = {p.Name.ToLower()};"))}}
            }

            public Snapshot CreateSnapshot() => new(this);
            ILayerSnapshot MachineLearning.Model.Layer.ILayer.CreateSnapshot() => CreateSnapshot();
            public Gradients CreateGradientAccumulator() => new(this);
            IGradients MachineLearning.Model.Layer.ILayer.CreateGradientAccumulator() => CreateGradientAccumulator();

            public long WeightCount => {{string.Join(" + ", weights.Select(p => IsVector(p.Type) ? $"{p.Name}.Count" : $"{p.Name}.FlatCount"))}};

            public sealed partial class Snapshot({{layer.Name}} layer) : ILayerSnapshot
            {
                // TODO: 
                // public {{tin.Name}} Input { get; } = {{tin.Name}}.Create(T, E);
                // public {{tin.Name}} GradientInput { get; } = {{tin.Name}}.Create(T, E);
                // public {{tout.Name}} Output { get; } = {{tout.Name}}.Create(T, E);
            }
        
        """);

        sb.AppendLine($"\tpublic sealed partial class Gradients({layer.Name} layer) : IGradients\n\t{{");

        foreach (var weight in weights)
        {
            sb.AppendLine($"\t\tpublic {weight.Type} {weight.Name} {{ get; }} = {weight.Type}.OfSize(layer.{weight.Name});");
        }

        sb.AppendLine($$"""
                public void Add(IGradients other)
                {
                    var o = Guard.Is<Gradients>(other);
        """);

        foreach (var weight in weights)
        {
            sb.AppendLine($"\t\t\t{weight.Name}.AddToSelf(o.{weight.Name});");
        }

        sb.AppendLine("\t\t}");

        sb.AppendLine($$"""
                public void Reset()
                {
        """);

        foreach (var weight in weights)
        {
            sb.AppendLine($"\t\t\t{weight.Name}.ResetZero();");
        }

        sb.AppendLine("\t\t}\n\t}");

        if (layer!.GetAttributes().FirstOrDefault(a => IsLayerSerializerAttribute(a.AttributeClass!)) is AttributeData ad)
        {
            sb.Insert(0, "using MachineLearning.Serialization;\n");
            sb.AppendLine($$"""
                public static partial class Serializer
                {
                    [System.Runtime.CompilerServices.ModuleInitializer]
                    internal static void Register()
                    {
                        MachineLearning.Serialization.ModelSerializer.RegisterLayer("{{ad.ConstructorArguments[0].Value}}", {{ad.ConstructorArguments[1].Value}}, Save, Read);
                    }

                    public static ErrorState Save({{layer.Name}} layer, System.IO.BinaryWriter writer)
                    {
                        {{string.Join("\n\t\t\t", parameter.Select(w => w.Type is { Name: "IActivationFunction" } ? $"ActivationFunctionSerializer.Write(writer, layer.{w.Name});" : $"writer.Write(layer.{w.Name});"))}}
                        {{string.Join("\n\t\t\t", weights.Select(w => $"ModelSerializationHelper.Write{w.Type.Name}(layer.{w.Name}, writer);"))}}
                        return default;
                    }

                    public static Result<{{layer.Name}}> Read(System.IO.BinaryReader reader)
                    {
                        return new {{layer.Name}}({{string.Join(", ", [.. parameter.Select(w => w.Type is { Name: "IActivationFunction" } ? $"ActivationFunctionSerializer.Read(reader)" : $"reader.Read{w.Type.Name}()"), .. weights.Select(w => $"ModelSerializationHelper.Read{w.Type.Name}(reader)")])}});
                    }
                }
            """);
        }

        sb.AppendLine("}");

        if (layer.GetAttributes().FirstOrDefault(a => IsGenerateOptimizersAttribute(a.AttributeClass!)) is AttributeData attributeData)
        {
            AdamLayerGenerator.GenerateAdam(context, compilation, new(layer, tin, tout, tsnap, weights), attributeData);
        }

        context.AddSource($"{layer.Name}.g.cs", sb.ToString());
    }
}
