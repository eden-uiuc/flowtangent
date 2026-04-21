from typing import Callable

def inputs(*dependencies: str):
    def decorator(func: Callable):
        func._inputs = set(dependencies)
        return func
    return decorator


def outputs(*outputs: str):
    def decorator(func: Callable):
        func._outputs = set(outputs)
        return func
    return decorator