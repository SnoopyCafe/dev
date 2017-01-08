
def null_decorator(func):
    return func

@null_decorator
def example():
    print('hello')




example = null_decorator(example)
