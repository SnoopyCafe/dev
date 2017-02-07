"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""

from miniflow import *

def compute(node, feed):
    # NOTE: because topological_sort set the values for the `Input` nodes we could also access
    # the value for x with x.value (same goes for y).
    sorted_nodes = topological_sort(feed_dict)
    output = forward_pass(node, sorted_nodes)

    outs = ""
    for inp in feed:
        outs = outs + "{" + str(inp.value) + "} "

    print (outs.replace(" ",node.operation,len(feed)-1)
       + " = {} (according to miniflow)".format(output))


#x, y, z = Input(), Input(), Input()
#feed_dict = {x: 10, y: 5, z: 8}

#compute(Add([x, y, z]), feed_dict)
#compute(Mult([x, y, z]), feed_dict)

inputs, weights, bias = Input(), Input(), Input()

f = Linear([inputs, weights, bias])

feed_dict = {
    inputs: [6, 14, 3],
    weights: [0.5, 0.25, 1.4],
    bias: 2
}
compute(f, feed_dict)

#print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))



