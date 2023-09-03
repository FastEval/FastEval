import asyncio
import multiprocessing.pool


async def process_with_progress_bar(*, items, process_fn, progress_bar_description):
    from tqdm.asyncio import tqdm_asyncio

    return await tqdm_asyncio.gather(
        *[process_fn(item) for item in items], desc=progress_bar_description
    )


async def join_tasks():
    while True:
        remaining_tasks = asyncio.all_tasks()
        remaining_tasks.remove(asyncio.current_task())
        if len(remaining_tasks) == 0:
            break
        await asyncio.wait(remaining_tasks)
