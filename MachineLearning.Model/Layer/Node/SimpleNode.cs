namespace MachineLearning.Model.Layer.Node;

public sealed class SimpleNode(int weightCount) : INode<double>
{
    public double[] Weights { get; } = new double[weightCount];
    public double Bias { get; set; }
}
