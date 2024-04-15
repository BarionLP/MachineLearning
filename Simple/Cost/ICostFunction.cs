namespace Simple;

public interface ICostFunction{
    public Number Cost(Number outputActivation, Number expected){
        var error = outputActivation - expected;
        return error * error;
    }

    public Number Derivative(Number outputActivation, Number expected){
        return 2 * (outputActivation - expected);
    }
}
