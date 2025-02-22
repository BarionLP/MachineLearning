namespace ML.Analyzer;

[Generator]
public sealed class AttributeGenerator : IIncrementalGenerator
{
    public void Initialize(IncrementalGeneratorInitializationContext context)
    {
        context.RegisterPostInitializationOutput(context =>
        {
            context.AddSource("Attributes.g.cs", "");
        });
    }
}