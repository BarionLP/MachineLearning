using System.Text;

namespace Simple.Training;

public sealed record DataPoint(Number[] Input, Number[] Expected) : DataPoint<Number[], Number[]>(Input, Expected) {
    public override string ToString() {
        var sb = new StringBuilder();
        sb.Append('[').AppendCollection(Input, ", ").Append("] => [").AppendCollection(Expected, ", ").Append(']');
        return sb.ToString();
    }
}
public record DataPoint<TInput, TExpected>(TInput Input, TExpected Expected);