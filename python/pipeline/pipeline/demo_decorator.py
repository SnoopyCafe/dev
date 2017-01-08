

# One of the most common uses of decorators is to add attributes to functions

def event_handler(func):
    func.is_event_handler = True
    return func

# It's possible to call multiple decorators; the one closest to the
# fcn will be called first, then the return value paassed to the next


def a(func):
    print('a')
    return func


def b(func):
    print('b')
    return func

@a
@b
@event_handler
def foo(event):
    pass


if getattr(foo,'is_event_handler',False):
    print("It's an event handler")

