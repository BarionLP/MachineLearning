using MachineLearning.Data;
using ML.MultiLayerPerceptron;

namespace MachineLearning.Samples;

public interface ISample<TInput, TOutput>
{
    public static abstract EmbeddedModel<TInput, TOutput> TrainDefault(EmbeddedModel<TInput, TOutput>? model = null, TrainingConfig? config = null, Random? random = null);
    public static abstract EmbeddedModel<TInput, TOutput> CreateModel(Random? random = null);
    public static abstract TrainingConfig DefaultTrainingConfig(Random? random = null);
    public static abstract ITrainingSet GetTrainingSet(Random? random = null);
}
