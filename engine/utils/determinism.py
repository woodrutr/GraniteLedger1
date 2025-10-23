from collections import OrderedDict

def ordered(m):
    return OrderedDict(sorted(m.items(), key=lambda kv: str(kv[0])))
