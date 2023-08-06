import multiprocessing.pool
import threading

def process_with_thread_pool(*, num_threads, items, process_fn, progress_bar_description=None):
    import tqdm

    def process_with_index(item_with_index):
        index, item = item_with_index
        result = process_fn(item)
        return index, result

    with multiprocessing.pool.ThreadPool(min(num_threads, len(items))) as pool:
        iterator = pool.imap_unordered(process_with_index, enumerate(items))
        results_with_indices = list(tqdm.tqdm(iterator, total=len(items), desc=progress_bar_description))

    return [result_with_index[1] for result_with_index in sorted(results_with_indices, key=lambda item: item[0])]

def join_threads():
    for thread in threading.enumerate():
        if thread.daemon:
            continue

        try:
            thread.join()
        except RuntimeError as error:
            if 'cannot join current thread' in error.args[0]: # main thread
                pass
            else:
                raise
