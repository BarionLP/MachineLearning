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
            sb.AppendLine($$"""using global::ML.Core.Modules;""");
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
                OutputModuleName => $$"""{{InputModuleName}}<{{moduleInfo.ArchType}}, {{moduleInfo.RootModule.TypeArguments[1]}}, {{moduleInfo.SnapshotTypeString}}, {{moduleInfo.GradientsTypeString}}>""",
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

        var parameterCount = string.Join(" + ", [.. moduleInfo.SubModules.Select(m => $"{m.Name}.ParameterCount"), .. moduleInfo.Weights.Select(w => $"(ulong){w.Name}.FlatCount")]);
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
            GenerateSnapshot(sb, moduleInfo);
            sb.AppendLine();
            GenerateGradients(sb, moduleInfo);
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
            GenerateSerializer(sb, moduleInfo);
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

    private static void GenerateSnapshot(StringBuilder sb, ModuleInfo moduleInfo)
    {
        sb.AppendLine($$"""
            public sealed partial class Snapshot({{moduleInfo.ModuleDefinitionString}} module) : IModuleSnapshot
            {
        """);

        foreach (var sub in moduleInfo.SubModules)
        {
            sb.AppendLine($$"""
                public {{sub.Module.SnapshotTypeString}} {{sub.Name}} { get; } = module.{{sub.Name}}.CreateSnapshot();
        """);
        }

        sb.AppendLine($$"""
        
                public void Dispose()
                {
        """);

        foreach (var sub in moduleInfo.SubModules)
        {
            sb.AppendLine($$"""
                    {{sub.Name}}.Dispose();
        """);
        }

        if (moduleInfo.Type.GetTypeMembers().FirstOrDefault(static t => t is { Name: "Snapshot" }) is { } snapshotType)
        {
            var fields = snapshotType.GetMembers().OfType<IFieldSymbol>().Where(static f => f.AssociatedSymbol is null && f.Type.AllInterfaces.Any(IsIDisposable));
            var properties = snapshotType.GetMembers().OfType<IPropertySymbol>().Where(static p => p.Type.AllInterfaces.Any(IsIDisposable));
            foreach (var field in fields)
            {
                sb.AppendLine($$"""
                        {{field.Name}}.Dispose();
            """);
            }
            foreach (var property in properties)
            {
                sb.AppendLine($$"""
                        {{property.Name}}.Dispose();
            """);
            }

            if (snapshotType.GetMembers().Any(m => m is IMethodSymbol { Name: "OnDispose" }))
            {
                sb.AppendLine("""
                        OnDispose();
            """);
            }
        }


        sb.AppendLine($$"""
                }
            }
        """);
    }

    private static void GenerateGradients(StringBuilder sb, ModuleInfo moduleInfo)
    {
        sb.AppendLine($$"""
            public sealed partial class Gradients({{moduleInfo.ModuleDefinitionString}} module) : IModuleGradients<Gradients>
            {
        """);

        foreach (var sub in moduleInfo.SubModules)
        {
            sb.AppendLine($$"""
                public {{sub.Module.GradientsTypeString}} {{sub.Name}} { get; } = module.{{sub.Name}}.CreateGradients();
        """);
        }

        foreach (var weight in moduleInfo.Weights)
        {
            sb.AppendLine($$"""
                public {{weight.Type}} {{weight.Name}} { get; } = {{weight.Type}}.OfSize(module.{{weight.Name}});
        """);
        }

        sb.AppendLine($$"""
        
                public void Add(Gradients other)
                {
        """);

        foreach (var module in moduleInfo.SubModules)
        {
            sb.AppendLine($$"""
                    {{module.Name}}.Add(other.{{module.Name}});
        """);
        }

        foreach (var weight in moduleInfo.Weights)
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

        foreach (var module in moduleInfo.SubModules)
        {
            sb.AppendLine($$"""
                    {{module.Name}}.Reset();
        """);
        }

        foreach (var weight in moduleInfo.Weights)
        {
            sb.AppendLine($$"""
                    {{weight.Name}}.ResetZero();
        """);
        }

        if (moduleInfo.Type.GetTypeMembers().FirstOrDefault(static t => t is { Name: "Gradients" }) is { } gradientsType)
        {
            if (gradientsType.GetMembers().Any(static m => m is IMethodSymbol { Name: "OnReset" }))
            {
                sb.AppendLine("""
                    OnReset();
        """);
            }
        }

        sb.AppendLine($$"""
                }
            }
        """);
    }

    private static void GenerateSerializer(StringBuilder sb, ModuleInfo module)
    {
        if (module.SubModules.Length is 0 && module.Weights.Length is 0 && module.Properties.Length is 0)
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

            foreach (var property in module.Properties)
            {
                sb.AppendLine($$"""
                if (!objectReader.TryRead{{property.Type.Name}}Value().Branch(out var {{property.Name}}, out error))
                {
                    return error;
                }
        """);
            }

            foreach (var subModule in module.SubModules)
            {
                sb.AppendLine($$"""
                if (!AmetrinSerializer.TryReadDynamic<{{subModule.Property.Type}}>(objectReader).Branch(out var {{subModule.Name}}, out error))
                {
                    return error;
                }
        """);
            }

            foreach (var weight in module.Weights)
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
                return new {{module.ModuleDefinitionString}}({{string.Join(", ", [.. module.Properties.Select(static p => p.Name), .. module.SubModules.Select(static m => m.Name), .. module.Weights.Select(static w => w.Name)])}});
            }

            public static void WriteValue(IAmetrinWriter writer, {{module.ModuleDefinitionString}} value) 
            {
                using var objectWriter = writer.WriteStartObject();

        """);

            foreach (var property in module.Properties)
            {
                sb.AppendLine($$"""
                objectWriter.Write{{property.Type.Name}}Value(value.{{property.Name}});
        """);
            }

            foreach (var subModule in module.SubModules)
            {
                sb.AppendLine($$"""
                AmetrinSerializer.WriteDynamic<{{subModule.Property.Type}}>(objectWriter, value.{{subModule.Name}});
        """);
            }

            foreach (var weight in module.Weights)
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
