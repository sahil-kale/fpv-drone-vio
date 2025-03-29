DEBUG_COUNTER = {}

def conditional_breakpoint(skip, key="default", single_shot=True):
    from pdb import set_trace
    count = DEBUG_COUNTER.get(key, 0) + 1
    DEBUG_COUNTER[key] = count
    if count > skip:
        set_trace()
