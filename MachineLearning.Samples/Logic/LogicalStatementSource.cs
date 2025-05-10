namespace MachineLearning.Samples.Logic;

public static class LogicalStatementSource
{
    public static IEnumerable<(string, char)> GenerateAdditionStatements(int count, Random? random = null)
    {
        random ??= Random.Shared;
        foreach (var a in 1..(count + 1))
        {
            foreach (var b in 1..(count + 1))
            {
                var result = a + b;
                var statement = $"{a}+{b}=";
                var resultString = $"{result}\0";

                foreach (var sub in resultString)
                {
                    yield return (statement, sub);
                    statement += sub;
                }
            }
        }
    }
}
