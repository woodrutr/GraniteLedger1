"""
set of validator functions for use in model
"""

import typing
import numpy as np


if typing.TYPE_CHECKING:
    from model.h2_model import H2Model


def region_validator(hm: 'H2Model', region):
    """checks if region name is string or numeric

        Parameters
        ----------
        hm : H2Model
            model
        region : any
            region name

    Raises:
        ValueError: region wrong type

    Returns:
        bool: is correct type
    """
    if isinstance(region, (str, int, np.int64)):
        return True
    raise ValueError(f'region name {region} is of type: {type(region)}')
