namespace MachineLearning.Training;

public static class ModelTrainer
{
    public static LegacyModelTrainer<TInput, TOutput> Legacy<TInput, TOutput>(EmbeddedModel<TInput, TOutput> model, TrainingConfig<TInput, TOutput> config) where TInput : notnull where TOutput : notnull
            => new(model, config);

    public static GenericModelTrainer<TInput, TOutput> Generic<TInput, TOutput>(IEmbeddedModel<TInput, TOutput> model, TrainingConfig<TInput, TOutput> config) where TInput : notnull where TOutput : notnull
            => new(model, config);
}