using MachineLearning.Data;
using MachineLearning.Data.Entry;
using MachineLearning.Training.Evaluation;

namespace MachineLearning.Training;

public interface ITrainer<TModel>
{
    public TModel Model { get; }
    public TrainingConfig Config { get; }
    public ITrainingSet TrainingSet { get; }
    DataSetEvaluationResult TrainAndEvaluate(IEnumerable<TrainingData> data);
    void FullReset();
}
