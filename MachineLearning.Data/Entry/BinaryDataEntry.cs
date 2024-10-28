using System.Text;

namespace MachineLearning.Data.Entry;

public sealed record BinaryDataEntry(float[] Input, bool Expected) : DataEntry<float[], bool>(Input, Expected)
{
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.Append('[').AppendJoin(", ", Input).Append("] => ").Append(Expected);
        return sb.ToString();
    }
}
