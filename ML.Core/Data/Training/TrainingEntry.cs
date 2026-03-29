namespace ML.Core.Data.Training;

public sealed record TrainingEntry<TInput, TArch, TExpected>(TInput InputValue, TArch ExpectedWeights, TExpected ExpectedValue);
