import sys
import shlex

def get_command():
    '''
    Get the full command that initiated the current process.
    '''
    return f"{sys.executable} " + " ".join(map(shlex.quote, sys.argv))