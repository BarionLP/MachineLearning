namespace ML.Analyzer;

internal sealed class MethodBodyWriter(StringBuilder sb, int indent = 0)
{
    private readonly StringBuilder sb = sb;

    public int Indent { get; set; } = indent;

    public void WriteOperation(string operation)
    {
        SbIndent();
        sb.AppendLine(operation);
    }

    public void OpenScope()
    {
        WriteOperation("{");
        Indent++;
    }

    public void CloseScope()
    {
        Indent--;
        WriteOperation("}");
    }
    
    private void SbIndent()
    {
        sb.Append(' ', Indent * 4);
    }
}
