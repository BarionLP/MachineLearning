namespace MachineLearning.Training;

/// <summary>
/// Converts data into expected output weights
/// </summary>
/// <typeparam name="TOutput">Network Output type</typeparam>
public interface IOutputResolver<in TOutput>
{
    public Vector Expected(TOutput output);
}
