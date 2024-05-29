import inspect

import numpy as np
from icecream import ic


def shape_name_short(arr, i, j):
    """
    Generates a shape name based on the dimensions of the input array.

    Args:
        arr (ndarray): The input array.
        i (int): The index of the first dimension to mark with 'X' in the shape name.
        j (int): The index of the second dimension to mark with 'X' in the shape name.

    Returns:
        str: The generated shape name.

    Example:
        >>> import numpy as np
        >>> arr = np.zeros((3, 4, 5))
        >>> shape_name_short(arr, 0, 2)
        'head_of_dims_XOX'
    """
    shape_name = ['O'] * arr.ndim
    shape_name[i] = 'X'  # f'>{arr.shape[i]}<'
    shape_name[j] = 'X'  # f'>{arr.shape[j]}<'
    shape_name = "".join(shape_name)
    shape_name = f'head_of_dims_{shape_name}'
    return shape_name

def shape_name_long(arr, i, j):
    """
    Returns a string representation of the shape of the input array with modified dimensions.

    Parameters:
    arr (ndarray): The input array.
    i (int): The index of the first dimension to modify.
    j (int): The index of the second dimension to modify.

    Returns:
    str: A string representation of the modified shape of the input array.

    Example:
    >>> import numpy as np

    >>> arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    >>> shape_name(arr_3d, 0, 2)
    '(>2<, 2, >2<)'
    """

    ith_dim, jth_dim = arr.shape[i], arr.shape[j]
    ith_name, jth_name = f'>{ith_dim}<', f'>{jth_dim}<'
    shape_name = []
    for k in range(arr.ndim):
        if k == i:
            shape_name.append(ith_name)
        elif k == j:
            shape_name.append(jth_name)
        else:
            shape_name.append(arr.shape[k])
    shape_name = ", ".join(map(str, shape_name))
    shape_name = f'({shape_name})'
    
    return shape_name

def get_head_by_dimension_pairs(arr, m=0, n=5, log=True):
    result = []
    for i in range(arr.ndim):
        for j in range(i+1, arr.ndim):            
            index = [0] * arr.ndim
            index[i] = slice(m, m+n)
            index[j] = slice(m, m+n)
            index = tuple(index)
            var_head = arr[index]
            
            # Print the shape highlighted by the two dimensions
            var_dimensions = shape_name_long(arr, i, j)
            
            # result[f'{i}x{j}'] = (shape_name, array_head)
            result.append(var_dimensions)
            result.append(var_head)
            
            if log: ic(var_dimensions, var_head)
    
    return result


class debug:
    """
    A class for debugging variables and their values.

    This class provides a convenient way to print the names and values of variables for debugging purposes.
    It supports printing numpy arrays with different dimensions.

    Args:
        *args: Variable arguments to be debugged.
        **kwargs: Keyword arguments to set the parameters of the debug instance.

    Keyword Args:
        first (int): The starting index for slicing arrays. Default is 0.
        last (int): The ending index for slicing arrays. Default is 5. If negative, all elements are included.

    Attributes:
        first (int): The starting index for slicing arrays.
        last (int): The ending index for slicing arrays.

    Example:
        debug(1, 2, 3, first=1, last=3)
        # Output:
        # var_name: 1, var_content: 2
        # var_name: 2, var_content: 3
    """

    def __call__(self, *args, **kwargs):
        
        n = 5
        
        frame = inspect.currentframe()
        frame = inspect.getouterframes(frame)[1]
        code_context = inspect.getframeinfo(frame[0]).code_context[0].strip()
        arg_names = code_context[code_context.find('(') + 1:-1].split(',')
        arg_names = [name.strip() for name in arg_names]
        
        for i, var in enumerate(args):
            var_name = arg_names[i]
            var_content = var
            # ic(var_name, var_content)
            
            if type(var_content) == np.ndarray:
                if var_content.ndim == 1:
                    ic(var_name, var_content)
                    continue
                elif var_content.ndim == 2:
                    var_head = var_content[self.first:, self.first:]
                    var_shape = var_content.shape
                    ic(var_name, var_shape, var_head)
                    continue
                else:
                    ic(var_name)
                    var_head = get_head_by_dimension_pairs(var_content, m=self.first, n=self.last)
                    # print(*var_head, sep='\n')
                    # ic(var_name, *var_head)
            else:
                ic(var_name, var_content)
    
    def __init__(self, *args, **kwargs):
        self.params(**kwargs)
        self.__call__(*args, **kwargs)
        
    def params(self, first=0, last=5):
        if first < 0:
            last = None
        self.first = first
        self.last = last
            
        

'''# Example usage
vmi = np.random.rand(3, 4, 5)
debug(vmi)
'''


import hashlib

import numpy as np


def hasharr(arr: np.ndarray):
    """
    Calculate the hash value of a numpy array.

    Parameters:
    arr (numpy.ndarray): The input array.

    Returns:
    str: The hash value of the array.

    """
    
    # If type is not numpy array, then we can assume that it has an underlying numpy array, so it has a .to_numpy() method
    if not isinstance(arr, np.ndarray):
        arr = arr.to_numpy()
    
    hash_value = hashlib.blake2b(arr.tobytes(), digest_size=20).hexdigest()
    # ic(hash_value)
    return hash_value

"""# Fix the random seed
np.random.seed(0)

# Create a random array
arr = np.random.rand(3, 4)

# Make a dataframe with the array
import pandas as pd
df = pd.DataFrame(arr)

print(hasharr(df))"""
