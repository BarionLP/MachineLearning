using MachineLearning.Training.Cost;

namespace MachineLearning.Training.Optimization;

// must be completely readonly, will be used multiple times
// state must go in the optimizer. this will be created at the start of each training
public interface IOptimizerConfig<TData>
{
    public Number LearningRate { get; }
    public ICostFunction CostFunction { get; }
    public IOptimizer<TData> CreateOptimizer();
}
