using MachineLearning.Data;

namespace MachineLearning.Training;

public static class ModelTrainer
{
    public static EmbeddedModelTrainer<TIn, TOut> Create<TIn, TOut>(EmbeddedModel<TIn, TOut> model, TrainingConfig config, ITrainingSet trainingSet)
            => new(model, config, trainingSet);
}