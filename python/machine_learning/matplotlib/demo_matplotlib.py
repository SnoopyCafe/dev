import matplotlib.pyplot as plt
import numpy as np

# a = np.linspace(0,10,100)
# b = np.exp(-a)
#
# plt.plot(a)
# plt.show()

"""
Simple demo of a horizontal bar chart.
"""
def barh_example():
    plt.rcdefaults()


    # Example data
    people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
    y_pos = np.arange(len(people))
    performance = 3 + 10 * np.random.rand(len(people))
    error = np.random.rand(len(people))

    plt.barh(y_pos, performance, xerr=error, align='center', alpha=0.4)
    plt.yticks(y_pos, people)
    plt.xlabel('Performance')
    plt.title('How fast do you want to go today?')

    plt.show()

    
barh_example()