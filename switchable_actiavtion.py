from keras.layers import Activation, LeakyReLU


def SwitchableActivation(activation_type, name=None, alpha=None):
    def _f(x):
        if activation_type == "relu":
            x = Activation('relu', name=name)(x) if name else Activation('relu')(x)
        elif activation_type == "leaky-relu":
            x = LeakyReLU(alpha=alpha, name=name)(x) if name else LeakyReLU(alpha=alpha)(x)
        else:
            raise Exception("Unknown Activation")
        return x

    return _f
