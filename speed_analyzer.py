import time
from dataclasses import dataclass
import pandas as pd


@dataclass
class Stats:
    name: str
    number_of_calls: int
    total_time: float
    avg_time: float


watchlist: dict[object, Stats] = dict()


def add_to_watchlist(func):
    key = object()
    watchlist[key] = Stats(func.__name__, 0, 0, 0)

    def timed_func(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        t = time.perf_counter() - start

        watchlist[key].number_of_calls += 1
        watchlist[key].total_time += t
        watchlist[key].avg_time = watchlist[key].total_time / watchlist[key].number_of_calls

        return result

    return timed_func


def get_stats() -> pd.DataFrame:
    results = pd.DataFrame(columns=["name", "total_time", "number_of_calls", "avg_time"])

    for stat in watchlist.values():
        results.loc[len(results)] = {
            "name": stat.name,
            "total_time": stat.total_time,
            "number_of_calls": stat.number_of_calls,
            "avg_time": stat.avg_time,
        }
    return results
