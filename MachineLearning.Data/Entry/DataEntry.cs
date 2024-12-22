namespace MachineLearning.Data.Entry;

public record DataEntry<TInput, TExpected>(TInput Input, TExpected Expected);

public record TrainingData<TInput, TExpected>(TInput InputValue, TExpected ExpectedValue, Vector ExpectedWeights) : TrainingData(ExpectedWeights);
public record TrainingData(Vector ExpectedWeights);