namespace Simple;

public static class NodeHelper {
    public static Number NodeCost(Number outputActivation, Number expected) {
        var error = outputActivation - expected;
        return error * error;
    }

    public static Number NodeCostDerivative(Number outputActivation, Number expected) {
        return 2 * (outputActivation - expected);
    }
}