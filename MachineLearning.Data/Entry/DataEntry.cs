namespace MachineLearning.Data.Entry;

public record DataEntry<TInput, TExpected>(TInput Input, TExpected Expected);