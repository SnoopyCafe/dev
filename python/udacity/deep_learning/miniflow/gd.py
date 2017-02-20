def gradient_descent_update(x, gradx, learning_rate):
    """
    Performs a gradient descent update.
    Remember the gradient is initially in the direction of steepest ascent
    so subtracting learning_rate * gradx from x turns it into steepest descent.
    """
    # TODO: Implement gradient descent.

    # Return the new value for x
    return x - learning_rate * gradx