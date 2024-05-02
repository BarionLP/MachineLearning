using Simple.Network.Layer;
using Simple.Training.Cost;

namespace Simple.Training.Optimization.Layer;

public interface ILayerOptimizer<TData> {
    public RecordingLayer Layer { get; }
    public ICostFunction CostFunction { get; }
    public void Update(TData[] nodeValues);
    public void Apply(IOptimizerConfig<TData> optimizer, int dataCounter);
    public void Reset();
    public Number[] CalculateOutputLayerNodeValues(Number[] expected);
    public Number[] CalculateHiddenLayerNodeValues(RecordingLayer oldLayer, Number[] oldNodeValues);
}
