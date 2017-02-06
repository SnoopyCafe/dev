"""
This script builds and runs a graph with miniflow.

There is no need to change anything to solve this quiz!

However, feel free to play with the network! Can you also
build a network that solves the equation below?

(x + y) + y
"""


#from python.udacity.deep_learning import miniflow
from miniflow import *

x, y, z = Input(), Input(), Input()

a = Add([x, y, z])
feed_dict = {x: 10, y: 5, z: 8}

sorted_nodes = topological_sort(feed_dict)
add_output = forward_pass(a, sorted_nodes)

# NOTE: because topological_sort set the values for the `Input` nodes we could also access
# the value for x with x.value (same goes for y).
outs = ""
for inp in feed_dict:
    outs = outs + "{" + str(inp.value) + "} "
    #total += inp.value

print ("add: " + outs.replace(" ","+",len(feed_dict)) + " = {} (according to miniflow)".format(add_output))

m = Mult([x, y, z])
outs = ""
for inp in feed_dict:
    outs = outs + "{" + str(inp.value) + "} "
    #total += inp.value
mult_output = forward_pass(m, sorted_nodes)

print ("mult: " + outs.replace(" ","+",len(feed_dict)) + " = {} (according to miniflow)".format(mult_output))

#print("{} + {} = {} (according to miniflow)".format(feed_dict[x], feed_dict[y], output))



