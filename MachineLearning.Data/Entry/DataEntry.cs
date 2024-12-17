namespace MachineLearning.Data.Entry;

public record DataEntry<TInput, TExpected>(TInput Input, TExpected Expected);

public record TrainingData<TInput, TExpected>(TInput InputValue, TExpected ExpectedValue, Vector Input, Vector Expected) : TrainingData(Input, Expected);
public record TrainingData(Vector Input, Vector Expected);