namespace ML.Core.Training.Data;

public sealed record TrainingEntry<TInput, TArch, TExpected>(TInput InputValue, TArch ExpectedWeights, TExpected ExpectedValue);
