namespace MachineLearning.Samples;

public interface ISample<TInput, TOutput>
{
    public static abstract EmbeddedModel<TInput, TOutput> TrainDefault(EmbeddedModel<TInput, TOutput>? model = null, TrainingConfig<TInput, TOutput>? trainingConfig = null, Random? random = null);
    public static abstract EmbeddedModel<TInput, TOutput> CreateModel(Random? random = null);
    public static abstract TrainingConfig<TInput, TOutput> DefaultTrainingConfig(Random? random = null);
    public static abstract IEnumerable<DataEntry<TInput, TOutput>> GetTrainingSet(Random? random = null);
}
