from .rdopt import get_res_rdopt

_network_factory = {'res': get_res_rdopt}


def get_network(cfg):
    arch = cfg.network
    get_model = _network_factory[arch]
    network = get_model()
    return network
