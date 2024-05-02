using System.Text;

namespace Simple.Training;

public sealed record BinaryDataPoint(Number[] Input, bool Expected) : DataPoint<Number[], bool>(Input, Expected) {
    public override string ToString() {
        var sb = new StringBuilder();
        sb.Append('[').AppendJoin(", ", Input).Append("] => ").Append(Expected);
        return sb.ToString();
    }
}

public record DataPoint<TInput, TExpected>(TInput Input, TExpected Expected);