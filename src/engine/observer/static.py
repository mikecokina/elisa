from queue import Empty


def queue_writer(*args):
    args_queue, orbital_motion, n_threads = args
    for idx, position in enumerate(orbital_motion):
        args_queue.put((idx, ) + position)
    for _ in range(n_threads):
        args_queue.put("TERMINATOR")


def worker(*args):
    args_queue, result_list, error_list, initial_sys_kwargs = args

    while True:
        # fixme: increase timeout
        try:
            xargs = args_queue.get(timeout=1)
        except Empty:
            continue

        if xargs == "TERMINATOR":
            break
