
from pyboolnet.file_exchange import bnet2primes


# Read the Boolean model
def read_boolean_model(filepath=''):
    try:
        boolean_model_tmp = bnet2primes(filepath)
        boolean_model = dict(sorted(boolean_model_tmp.items()))     # Sort the nodes
        return boolean_model

    except RuntimeError as error:
        print(error)
        print('Could not read the Boolean model!')
