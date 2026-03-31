namespace ML.Core.Data.Training;

public interface ITrainingDataSource<T>
{
    public int BatchCount { get; }
    public int BatchSize { get; }
    public IEnumerable<IEnumerable<T>> GetBatches();
    public void Reset();
}
