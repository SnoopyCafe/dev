"""Run a sequence of programs.
Each program (except the first) receives the stand out of the
previous program on it statndard input, by default.
"""

import argparse

def __launch():
    parser = argparse.ArgumentParser(prog ='python -m pipeline',
                                     description = __doc__)

    parser.add_argument('-k','--keep-going',
                        action = 'store_true',
                        default = False)
    parser.add_argument('filename')
    args = parser.parse_args()
    print(args.keep_going)
    print(args.filename)

if __name__ == '__main__':
    __launch()