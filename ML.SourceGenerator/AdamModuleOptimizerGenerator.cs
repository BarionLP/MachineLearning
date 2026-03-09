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

        foreach (var submodule in moduleInfo.Modules)
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

        foreach (var submodule in moduleInfo.Modules)
        {
            sb.AppendLine($$"""
                {{submodule.Name}}Optimizer.Apply(gradients.{{submodule.Name}});
        """);
        }


        foreach (var weight in moduleInfo.Weights)
        {
            sb.AppendLine($$"""
        
                (FirstMoment{{weight.Name}}, gradients.{{weight.Name}}).MapToFirst(Optimizer.FirstMomentEstimate, Optimizer.FirstMomentEstimate);
                NumericsDebug.AssertValidNumbers(FirstMoment{{weight.Name}});
                (SecondMoment{{weight.Name}}, gradients.{{weight.Name}}).MapToFirst(Optimizer.SecondMomentEstimate, Optimizer.SecondMomentEstimate);
                NumericsDebug.AssertValidNumbers(SecondMoment{{weight.Name}});
                Module.{{weight.Name}}.SubtractToSelf((FirstMoment{{weight.Name}}, SecondMoment{{weight.Name}}).Map(Optimizer.WeightReduction, Optimizer.WeightReduction));
        """);
        }
        
        sb.AppendLine($$"""
            }

            public void FullReset()
            {
        """);

        foreach (var submodule in moduleInfo.Modules)
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
