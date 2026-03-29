using System.Text;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace ML.SourceGenerator;

[Generator]
internal sealed class AdamModuleOptimizerGenerator : IIncrementalGenerator
{
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        var optimizers = context.SyntaxProvider.ForAttributeWithMetadataName("ML.Core.Attributes.GeneratedAdamAttribute",
            (node, _) => node is ClassDeclarationSyntax,
            (context, token) => context.SemanticModel.GetDeclaredSymbol(context.TargetNode, token) as INamedTypeSymbol
        );

        context.RegisterSourceOutput(optimizers.Combine(context.CompilationProvider), GenerateAdam);
    }

    private static void GenerateAdam(SourceProductionContext context, (INamedTypeSymbol?, Compilation) pair)
    {
        var (optimizer, compilation) = pair;
        if (optimizer is null) return;

        var attribute = optimizer.TryGetAttribute(IsGeneratedAdamAttribute);
        Debug.Assert(attribute is not null);

        if (attribute.ConstructorArguments[0].Value is not INamedTypeSymbol module) return;

        var moduleInfo = SubModuleInfo.CreateFull(module, canGenerateDataClasses: true);
        if (moduleInfo is null) return;

        var sb = new StringBuilder();

        sb.AppendLine($$"""
        using ML.Core.Training;
        """);

        var containerCount = BuildFileHeaderFor(optimizer, sb);


        BuildTypeHeader(optimizer, sb);


        sb.AppendLine($$"""
        (AdamOptimizer optimizer, {{moduleInfo.ModuleDefinitionString}} module) : IModuleOptimizer<{{moduleInfo.GradientsTypeString}}>
        {
        """);

        sb.AppendLine($$"""
            public {{moduleInfo.ModuleDefinitionString}} Module { get; } = module;
            public AdamOptimizer Optimizer { get; } = optimizer;

        """);

        foreach (var submodule in moduleInfo.SubModules)
        {
            sb.AppendLine($$"""
            public IModuleOptimizer {{submodule.Name}}Optimizer { get; } = optimizer.CreateModuleOptimizer(module.{{submodule.Name}});
        """);
        }

        foreach (var weight in moduleInfo.Weights)
        {
            sb.AppendLine($$"""

            public {{weight.Type}} FirstMoment{{weight.Name}} { get; } = {{weight.Type}}.OfSize(module.{{weight.Name}});
            public {{weight.Type}} SecondMoment{{weight.Name}} { get; } = {{weight.Type}}.OfSize(module.{{weight.Name}});
        """);
        }

        sb.AppendLine($$"""
        
            public void Apply({{moduleInfo.GradientsTypeString}} gradients)
            {
        """);

        foreach (var submodule in moduleInfo.SubModules)
        {
            sb.AppendLine($$"""
                {{submodule.Name}}Optimizer.Apply(gradients.{{submodule.Name}});
        """);
        }

        if(moduleInfo.Weights.Length > 0)
        {
            sb.AppendLine($$"""
                var firstMomentEstimateOperation = Optimizer.FirstMomentEstimateOperation;
                var secondMomentEstimateOperation = Optimizer.SecondMomentEstimateOperation;
                var weightReductionOperation = Optimizer.WeightReductionOperation;
        """);


            foreach (var weight in moduleInfo.Weights)
            {
                sb.AppendLine($$"""
            
                SpanOperations.MapTo(in firstMomentEstimateOperation, FirstMoment{{weight.Name}}.AsSpan(), gradients.{{weight.Name}}.AsSpan(), FirstMoment{{weight.Name}}.AsSpan());
                NumericsDebug.AssertValidNumbers(FirstMoment{{weight.Name}});
                SpanOperations.MapTo(in secondMomentEstimateOperation, SecondMoment{{weight.Name}}.AsSpan(), gradients.{{weight.Name}}.AsSpan(), SecondMoment{{weight.Name}}.AsSpan());
                NumericsDebug.AssertValidNumbers(SecondMoment{{weight.Name}});
                SpanOperations.MapTo(in weightReductionOperation, Module.{{weight.Name}}.AsSpan(), FirstMoment{{weight.Name}}.AsSpan(), SecondMoment{{weight.Name}}.AsSpan(), Module.{{weight.Name}}.AsSpan());
        """);
            }
        }


        
        sb.AppendLine($$"""
            }

            public void FullReset()
            {
        """);

        foreach (var submodule in moduleInfo.SubModules)
        {
            sb.AppendLine($$"""
                {{submodule.Name}}Optimizer.FullReset();
        """);
        }


        foreach (var weight in moduleInfo.Weights)
        {
            sb.AppendLine($$"""
        
                FirstMoment{{weight.Name}}.ResetZero();
                SecondMoment{{weight.Name}}.ResetZero();
        """);
        }

        sb.AppendLine($$"""
            }

            [System.Runtime.CompilerServices.ModuleInitializer]
            internal static void Register()
            {
                global::ML.Core.Training.AdamOptimizer.Registry.Register<{{moduleInfo.ModuleDefinitionString}}>(static (op, module) => new {{optimizer.Name}}(op, module));
            }
        """);


        foreach (var _ in Enumerable.Range(0, containerCount + 1))
        {
            sb.AppendLine("}");
        }


        context.AddSource($"{module.Name}.Adam.g.cs", sb.ToString());
    }
}
