from main import *

def test_evaluation():
    expressions, ground_truth = [], []
    
    expressions.append(If(LessThan(NumberVariable("x"), NumberVariable("y")),
                          NumberVariable("x"),
                          NumberVariable("y")))
    ground_truth.append(lambda x, y: min(x,y))
    
    expressions.append(If(Not(LessThan(NumberVariable("x"), NumberVariable("y"))),
                          Times(NumberVariable("x"), NumberVariable("y")),
                          Plus(NumberVariable("x"), NumberVariable("y"))))
    ground_truth.append(lambda x, y: x * y if not (x < y) else x + y)

    expressions.append(Times(NumberVariable("x"), Plus(NumberVariable("y"), Number(5))))
    ground_truth.append(lambda x, y: x * (y + 5))

    expressions.append(FALSE())
    ground_truth.append(lambda x, y: False)
    
    expressions.append(Not(FALSE()))
    ground_truth.append(lambda x, y: True)

    all_correct = True
    for expression, correct_semantics in zip(expressions, ground_truth):
        this_correct = True
        for x in range(10):
            for y in range(10):
                if expression.evaluate({"x": x, "y": y}) != correct_semantics(x,y):
                    this_correct = False
        if not this_correct:
            print("problem with evaluation for expression:")
            print(expression)
            print("please debug `evaluate` methods")
        all_correct = all_correct and this_correct

    if all_correct:
        print(" [+] arithmetic evaluation passes checks")


if __name__ == "__main__":
    test_evaluation()
