from typing import Callable

def inputs(*dependencies: str):
    def decorator(func: Callable):
        func._inputs = set(dependencies)
        return func
    return decorator


def outputs(*results: str):
    def decorator(func: Callable):
        func._results = set(results)
        return func
    return decorator