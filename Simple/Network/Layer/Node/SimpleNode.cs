namespace Simple.Network.Layer.Node;

public sealed class SimpleNode(int weightCount) : INode<Number> {
    public Number[] Weights { get; } = new Number[weightCount];
    public Number Bias { get; set; }
}
