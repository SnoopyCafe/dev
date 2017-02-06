
def test_takes_any_args(*args):
    message = "Input args:"
    for arg in args:
        message += str(arg) + " "
    return message


print (test_takes_any_args("x","y", "z"))