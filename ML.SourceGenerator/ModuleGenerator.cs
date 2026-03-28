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

        var attribute = module.TryGetAttribute(IsGeneratedModuleAttribute);
        Debug.Assert(attribute is not null);

        var includeSerializer = (bool)attribute.ConstructorArguments[0].Value!;

        var moduleInfo = SubModuleInfo.CreateFull(module, canGenerateDataClasses: true);
        Debug.Assert(moduleInfo is not null);

        var sb = new StringBuilder();

        if (includeSerializer)
        {
            sb.AppendLine($$"""using global::Ametrin.Serializer;""");
            sb.AppendLine($$"""using global::ML.Core.Converters;""");
            sb.AppendLine();
            sb.AppendLine();
        }

        var containerCount = BuildFileHeaderFor(module, sb);

        sb.Append($$"""
        partial class {{moduleInfo.ModuleDefinitionString}}
        """);

        if (moduleInfo.GenerateDataClasses || includeSerializer)
        {
            sb.Append($$""" : """);
        }

        if (moduleInfo.GenerateDataClasses)
        {
            sb.Append(moduleInfo.RootModule.Name switch
            {
                ModuleName => $$"""{{ModuleName}}<{{moduleInfo.ArchType}}, {{moduleInfo.SnapshotTypeString}}, {{moduleInfo.GradientsTypeString}}>""",
                HiddenModuleName => $$"""{{HiddenModuleName}}<{{moduleInfo.ArchType}}, {{moduleInfo.SnapshotTypeString}}, {{moduleInfo.GradientsTypeString}}>""",
                InputModuleName => $$"""{{InputModuleName}}<{{moduleInfo.RootModule.TypeArguments[0]}}, {{moduleInfo.ArchType}}, {{moduleInfo.SnapshotTypeString}}, {{moduleInfo.GradientsTypeString}}>""",
                _ => throw new NotImplementedException($"cannot impl {moduleInfo.RootModule.Name}"),
            });
        }

        if (includeSerializer)
        {
            if (moduleInfo.GenerateDataClasses)
            {
                sb.Append($$""", """);
            }
            sb.Append($$"""ISerializationConverter<{{moduleInfo.ModuleDefinitionString}}>""");
        }

        sb.AppendLine();

        var parameterCount = string.Join(" + ", [.. moduleInfo.Modules.Select(m => $"{m.Name}.ParameterCount"), .. moduleInfo.Weights.Select(w => $"(ulong){w.Name}.FlatCount")]);
        if (string.IsNullOrEmpty(parameterCount))
        {
            parameterCount = "0";
        }
        sb.AppendLine($$"""
        {
            public ulong ParameterCount => {{parameterCount}};

            public {{moduleInfo.SnapshotTypeString}} CreateSnapshot() => {{(IsEmptyModuleData(moduleInfo.SnapshotType) ? "global::ML.Core.Modules.EmptyModuleData.Instance" : "new(this)")}};
            public {{moduleInfo.GradientsTypeString}} CreateGradients() => {{(IsEmptyModuleData(moduleInfo.GradientsType) ? "global::ML.Core.Modules.EmptyModuleData.Instance" : "new(this)")}};
        """);

        if (moduleInfo.GenerateDataClasses)
        {
            sb.AppendLine();
            GenerateSnapshot(sb, moduleInfo.ModuleDefinitionString, moduleInfo.Modules, moduleInfo.Weights);
            sb.AppendLine();
            GenerateGradients(sb, moduleInfo.ModuleDefinitionString, moduleInfo.Modules, moduleInfo.Weights);
        }

        if (IsEmptyModuleData(moduleInfo.GradientsType))
        {
            sb.AppendLine($$"""

            [global::System.Runtime.CompilerServices.ModuleInitializer]
            internal static void RegisterOptimizer()
            {
                global::ML.Core.Training.AdamOptimizer.Registry.RegisterEmpty<{{moduleInfo.ModuleDefinitionString}}>();
            }
        """);
        }

        if (includeSerializer)
        {
            sb.AppendLine();
            GenerateSerializer(sb, moduleInfo, moduleInfo.Modules, moduleInfo.Weights);
        }


        sb.AppendLine($$"""
        }
        """);

        foreach (var _ in Enumerable.Range(0, containerCount))
        {
            sb.AppendLine("}");
        }

        context.AddSource($"{module.Name}.g.cs", sb.ToString());
    }

    private static void GenerateSnapshot(StringBuilder sb, string moduleDefinitionString, IEnumerable<ModulePropertyInfo> modules, IEnumerable<IPropertySymbol> weights)
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
        
                public void Dispose() 
                {
        """);

        foreach (var sub in modules)
        {
            sb.AppendLine($$"""
                    {{sub.Name}}.Dispose();
        """);
        }

        sb.AppendLine($$"""
                }
            }
        """);
    }

    private static void GenerateGradients(StringBuilder sb, string moduleDefinitionString, IEnumerable<ModulePropertyInfo> modules, IEnumerable<IPropertySymbol> weights)
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

    private static void GenerateSerializer(StringBuilder sb, ModuleInfo module, ImmutableArray<ModulePropertyInfo> modules, ImmutableArray<IPropertySymbol> weights)
    {
        if (modules.Length is 0 && weights.Length is 0)
        {
            sb.AppendLine($$"""
            public static Result<{{module.ModuleDefinitionString}}, DeserializationError> TryReadValue(IAmetrinReader reader) => Instance;
            public static void WriteValue(IAmetrinWriter writer, {{module.ModuleDefinitionString}} value) { }
        """);
        }
        else
        {
            sb.AppendLine($$"""
            public static Result<{{module.ModuleDefinitionString}}, DeserializationError> TryReadValue(IAmetrinReader reader)
            {
                using var objectReader = reader.ReadStartObject();
                DeserializationError error = default;
        
        """);

            foreach (var subModule in modules)
            {
                sb.AppendLine($$"""
                if (!AmetrinSerializer.TryReadDynamic<{{subModule.Property.Type}}>(objectReader).Branch(out var {{subModule.Name}}, out error))
                {
                    return error;
                }
        """);
            }

            foreach (var weight in weights)
            {
                sb.AppendLine($$"""
                if (!{{weight.Type.Name}}Converter.TryReadValue(objectReader).Branch(out var {{weight.Name}}, out error))
                {
                    return error;
                }
        """);
            }

            sb.AppendLine($$"""

                reader.ReadEndObject();
                return new {{module.ModuleDefinitionString}}({{string.Join(", ", modules.Select(static m => m.Name).Concat(weights.Select(static w => w.Name)))}});
            }

            public static void WriteValue(IAmetrinWriter writer, {{module.ModuleDefinitionString}} value) 
            {
                using var objectWriter = writer.WriteStartObject();

        """);

            foreach (var subModule in modules)
            {
                sb.AppendLine($$"""
                AmetrinSerializer.WriteDynamic<{{subModule.Property.Type}}>(objectWriter, value.{{subModule.Name}});
        """);
            }

            foreach (var weight in weights)
            {
                sb.AppendLine($$"""
                {{weight.Type.Name}}Converter.WriteValue(objectWriter, value.{{weight.Name}});
        """);
            }

            sb.AppendLine($$"""

                writer.WriteEndObject();
            }
        """);
        }

        if (!module.Type.IsGenericType)
        {
            sb.AppendLine($$"""

            [global::System.Runtime.CompilerServices.ModuleInitializer]
            internal static void RegisterSerializer()
            {
                AmetrinSerializer.RegisterSerializer<{{module.ModuleDefinitionString}}, {{module.ModuleDefinitionString}}>();
            }
        """);
        }
    }
}
