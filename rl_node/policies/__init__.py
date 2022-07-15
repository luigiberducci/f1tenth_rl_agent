from .SACPolicy import SACMlpPolicy

register = {}


def __getattr__(name):
    return register[name]


def register_policy(name, cls):
    register[name] = cls


register_policy("SACMlpPolicy", SACMlpPolicy)
