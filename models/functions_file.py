import time

# this file can be used for general functions that will be used repeatedly (can just help to speed things up)

# also the way decoraters work (putting the @ symbol before a function) basically makes the function run inside of the decorator if that makes sense
def get_runtime(func):
    def wrapper(*args, **kwargs):
        init = time.time()
        func(*args, **kwargs)
        print(f"The function: {func.__name__} took {'%.3f'%(time.time() - init)} seconds to run.")
    return wrapper
# like here ive defined a wrapper (the function i want to add)
# in this case its a simple runtime function as it gets the time before the function runs
# then it runs the function and prints the runtime after
# idk what else they could be used for in this case but this one will be useful for optimising based off computation time
