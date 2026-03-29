namespace ML.SourceGenerator;

[Generator]
public sealed class GlobalUsingsGenerator : IIncrementalGenerator
{
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        context.RegisterPostInitializationOutput(static ctx =>
        {
            ctx.AddSource("GlobalUsings.g.cs", """
            global using Ametrin.Optional;
            global using Ametrin.Utils;
            global using Ametrin.Guards;
            global using Ametrin.Numerics;
            
            global using System;
            global using System.Collections.Generic;
            global using System.Collections.Frozen;
            global using System.Collections.Immutable;
            global using System.Diagnostics;
            global using System.Linq;
            global using System.Threading.Tasks;

            global using ML.Core;
            global using Weight = float;
            global using SimdVector = System.Numerics.Vector<float>;
            global using SimdVectorHelper = System.Numerics.Vector;
            """);
        });
    }
}
