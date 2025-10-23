"""Establish a base model class for the sectoral modules to inherit."""

###################################################################################################
# Setup
import pandas as pd
from typing import Literal, List, Tuple, Sequence, MutableSequence, Dict, DefaultDict, Set
from collections import defaultdict
import sys as sys
import pyomo.environ as pyo
import numpy as np
from logging import getLogger

# Establish logger
logger = getLogger(__name__)

###################################################################################################
# MODEL


class Model(pyo.ConcreteModel):
    """This is the base model class for the models.

    This class contains methods for declaring pyomo components, extracting duals, and
    decorating expressions. The model class methods and attributes provide functionality
    for keeping track of index labels and ordering for all pyomo components; this is
    essential for integration tasks without the use of hard-coded indices and allows for
    easy post-processing tasks.
    """

    def __init__(self, *args, **kwargs):
        pyo.ConcreteModel.__init__(self, *args, **kwargs)

        # Initialize decorators with model instance
        self.DefaultDecorator(self)

        # dictionary to lookup column names for sets, params, variables
        self.cols_dict = {}
        self.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    @classmethod
    def build(cls):
        """Default build command; class-level build to create and return an instance of Model.

        This will work for any class inheriting the method, but it is recommended to replace this
        with model-specific build instructions if this functionality is desired.

        Returns
        -------
        Object
            Instance of Model object
        """
        return cls()

    def reorganize_index_set(
        self,
        sname: str,
        new_sname: str,
        return_set: bool | None = False,
        create_indexed_set: bool | None = False,
        reorg_set_cols: List[str] | None = None,
        reorg_set_sname: str | None = None,
    ):
        """Creates new pyomo sets based on an input set and a desired set of indices for an output
        set. User should provide either names of columns desired for reorganized output set OR the
        name of a set that mirrors the desired indexing.

        For instance, an input set indexed by (yr, region, month, day) can be reorganized into an
        output set:

        (yr, region):[(month,day), (month,day), (month,day)]

        when ["yr", "region"] is provided for reorg_set_cols.

        If only the set keys are desired, without creating an indexed set object as illustrated
        above, the user can set 'create_indexed_set' to false. If true, the output is a
        pyo.IndexedSet, with each element of the IndexedSet containing the values of other indices

        Parameters
        ----------
        sname : str
            Name of input set
        new_sname : str
            Name of output set or IndexedSet
        create_indexed_set : bool | None, optional
            Indicator for whether output set should include values as well as new index (IndexedSets), by default False
        return_set: bool | None, optional
            Indicator for whether to return the constructed set
        reorg_set_cols : List[str] | None, optional
            List of columns to index output set contained in 'sname', by default None
        reorg_set_sname : str | None, optional
            Name of set to use for identifying output set indices, by default None

        Returns
        -------
        Pyo.Set
            Pyomo Set or IndexedSet object reorganized based on input set

        Raises
        ------
        ValueError
            Populate function is either-or for reorg_set_cols and reorg_set_sname, received both
        ValueError
           Populate function is either-or for reorg_set_cols and reorg_set_sname, received neither
        ValueError
            Elements missing from input set desired in new set
        """
        # reorganize_index_set -- Get a hook to the input set
        set_in = getattr(self, sname)
        set_in_cols = self.cols_dict[sname]

        # reorganize_index_set -- Throw error if both reorg options provided
        if reorg_set_cols and reorg_set_sname:
            raise ValueError(
                f'Populate function is either-or for reorg_set_cols and reorg_set_sname, received both: {reorg_set_cols}, {reorg_set_sname}'
            )
        elif not reorg_set_cols and not reorg_set_sname:
            raise ValueError(
                'Populate function is either-or for reorg_set_cols and reorg_set_sname; need to provide a template or names for new set or indexedset'
            )

        # reorganize_index_set --
        if reorg_set_sname:
            reorg_set = getattr(
                self, reorg_set_sname
            )  # placeholder for a function that crosschecks the indices if desired
            reorg_set_cols = self.cols_dict[reorg_set_sname]

        # reorganize_index_set -- Check to ensure names desired are included in the reorg set (or columns)
        missing_elements = set(reorg_set_cols) - set(set_in_cols)
        if bool(missing_elements):
            raise ValueError(
                f'Elements missing from input set desired in new set: {missing_elements}'
            )

        set_out_key_index = set([set_in_cols.index(x) for x in reorg_set_cols])
        set_out_val_index = set(
            [set_in_cols.index(x) for x in set_in_cols if x not in reorg_set_cols]
        )

        # build dictionary result of the indexed set from the input set
        self.cols_dict[new_sname] = reorg_set_cols
        res = defaultdict(list)
        for element in set_in:
            idx = tuple(element[t] for t in set_out_key_index)
            data = tuple(element[t] for t in set_out_val_index)
            res[idx].append(data)

        # make the indexed set from keys -> index set, data = the dictionary

        # Update cols_dict
        set_out = pyo.Set(initialize=res.keys())
        if create_indexed_set:
            set_out = pyo.Set(set_out, initialize=res)

        if not return_set:
            setattr(self, new_sname, set_out)
            return None
        return set_out

    def declare_set(
        self,
        sname: str,
        sdata: MutableSequence | pd.DataFrame | pd.Series | Dict,
        scols: MutableSequence | None = None,
        return_set: bool | None = False,
        switch: bool | None = True,
        create_indexed_set: bool | None = True,
        use_values: bool | None = False,
        use_columns: bool | None = False,
    ):
        """Declares a pyomo Set object named 'sname' using input index values and labels.

        Function handles input values and calls appropriate declare_set methods based on data
        type of sdata

        Parameters
        ----------
        sname : str
            Name of set
        sdata : Dict
            Data object that contains set values
        scols : Sequence | None, optional
            List of column names corresponding to index labels and position, by default None
        return_set : bool | None, optional
            Return the set rather than assign within function, by default False
        switch : bool | None, optional
            Return None if False, by default True
        create_indexed_set : bool | None, optional
            If dict, indicator for whether output set should include values as well as new index (IndexedSets), by default True
        use_values : bool | None, optional
            If dict and create_indexed_set is False, use the values of sdata rather than keys for pyo Set members, by default False
        use_columns: bool | None, optional
            If Pandas, use columns as indices for pyo set rather than row index, by default False

        Returns
        -------
        pyo.Set
            Pyomo Set Object
        """

        # declare_set -- Based on instance of sdata, call appropriate function and pass to sub-methods
        if isinstance(sdata, (pd.DataFrame, pd.Series)):
            return self._declare_set_with_pandas(
                sname, sdata, return_set=return_set, switch=switch, use_columns=use_columns
            )
        elif isinstance(sdata, (list, tuple, range, np.ndarray)):
            return self._declare_set_with_iterable(
                sname,
                sdata,
                scols=scols,
                return_set=return_set,
                switch=switch,
            )
        elif isinstance(sdata, (dict, defaultdict)):
            if scols is None:
                raise ValueError(
                    'Provided dictionary data without index labels; specify labels in scols'
                )
            return self._declare_set_with_dict(
                sname,
                sdata,
                scols,
                return_set=return_set,
                switch=switch,
                create_indexed_set=create_indexed_set,
            )
        else:
            raise ValueError(f"Object type {type(sdata)} not supported in 'declare_set'")

    def _declare_set_with_pandas(
        self,
        sname: str,
        sdata: pd.DataFrame | pd.Series,
        return_set: bool | None = False,
        switch: bool | None = True,
        use_columns: bool | None = False,
    ):
        """Declares a pyomo Set object named 'sname' using input index values and labels from a
        Pandas object.

        Function assumes that the index values are the desired data to construct set object. User
        can specify whether to create set with column values instead

        Parameters
        ----------
        sname : str
            Name of set
        sdata : MutableSequence | dict
            Data object that contains set indices
        return_set : bool | None, optional
            Return the set rather than assign within function, by default False
        switch : bool | None, optional
            Return None if False, by default True
        use_columns: bool | None, optional
            Use columns as indices for pyo set rather than row index, by default False

        Returns
        -------
        pyo.Set
            Pyomo Set Object
        """
        # declare_set -- If switch off, return empty set
        if not switch:
            return None

        if use_columns:
            sset = pyo.Set(initialize=sdata.values.tolist())
            scols = sdata.columns.to_list()
        else:
            sset = pyo.Set(initialize=sdata.index)
            scols = list(sdata.index.names)
            for label in scols:
                if not label:
                    raise ValueError(f'Unnamed index in {sname} pandas object; check names of indices in input DataFrame or Series \
                    or set use_columns = True to create set with columnar data (and column names as index labels)')

        # declare_set -- update cols_dict with column names
        self.cols_dict[sname] = scols

        # declare_set -- if return not desired, assign to self directly
        if not return_set:
            setattr(self, sname, sset)
            return None
        return sset

    def _declare_set_with_iterable(
        self,
        sname: str,
        sdata: Sequence | Set | np.array,
        scols: Sequence[str] | None = None,
        return_set: bool | None = False,
        switch: bool | None = True,
    ) -> pyo.Set:
        """Declares a pyomo Set object named 'sname' using input index values and labels.

        Function can take iterable objects such as tuples, lists, etc as data inputs. Note that if
        the dimension of the index is larger than 1, user needs to provide a list of names for each
        set dimension.

        Parameters
        ----------
        sname : str
            Name of set
        sdata : Sequence
            Data object that contains set values
        scols : Sequence | None, optional
            List of column names corresponding to index labels and position, by default None
        return_set : bool | None, optional
            Return the set rather than assign within function, by default False
        switch : bool | None, optional
            Return None if False, by default True

        Returns
        -------
        pyo.Set
            Pyomo Set Object
        """
        # declare_set_iterable -- If switch off , return None
        if not switch:
            return None

        # declare_set_iterable -- Check if empty, if yes, pass and return with warrnings
        if len(sdata) == 0:
            empty = True
            scols = sname
            logger.warning(
                'For %s, sdata: %s is length 0; returning empty set called %s',
                sname,
                type(sdata),
                sname,
            )
        else:
            empty = False

        # declare_set_with_iterable -- convert to list format if numpy array or set
        if isinstance(sdata, np.ndarray):
            sdata = sdata.tolist()
        elif isinstance(sdata, set):
            sdata = list(sdata)

        # declare_set_with_iterable -- initialize pyo set
        sset = pyo.Set(initialize=sdata)

        # declare_set_with_iterable -- check to ensure column labels exist before assigning to cols_dict
        if not scols and not empty:
            if isinstance(sdata[0], (list, tuple)):
                if len(sdata[0]) > 1:
                    raise ValueError(
                        'if using list or set w/ multiple dimensional index, need to provide labels of indices'
                    )
            else:
                logger.info(
                    f'declare_set_with_iterable assumes desired index name is sname = {sname} when instantiating 1-D set'
                )
                scols = [sname]

        # fix input if single string
        if isinstance(scols, str):
            scols = [scols]

        # declare_set_with_iterable -- update cols_dict with column names
        self.cols_dict[sname] = scols

        # declare_set -- if return not desired, assign to self directly
        if not return_set:
            setattr(self, sname, sset)
            return None
        return sset

    def _declare_set_with_dict(
        self,
        sname: str,
        sdata: Dict | DefaultDict,
        scols: MutableSequence,
        return_set: bool | None = False,
        switch: bool | None = True,
        create_indexed_set: bool | None = True,
        use_values: bool | None = False,
    ) -> pyo.Set:
        """Declares a pyomo Set object named 'sname' using input index values and labels.

        Function takes a dictionary argument and creates pyomo set object from keys, values, or both.

        If an indexed set is desired, set create_indexed_set to True; the function will create an
        indexed set with its own indices set as keys. Otherwise, an Ordered Scalar Set will be
        created, either from the keys or the values of 'sdata' depending on the value for
        'use_values' (False for keys, True for values).

        Names for the indices handled by scols; user must provide.

        Parameters
        ----------
        sname : str
            Name of set
        sdata : Dict
            Data object that contains set values
        scols : Sequence | None, optional
            List of column names corresponding to index labels and position, by default None
        return_set : bool | None, optional
            Return the set rather than assign within function, by default False
        switch : bool | None, optional
            Return None if False, by default True
        create_indexed_set : bool | None, optional
            Indicator for whether output set should include values as well as new index (IndexedSets), by default True
        use_values : bool | None, optional
            If create_indexed_set is False, use the values of sdata rather than keys for pyo Set members, by default False

        Returns
        -------
        pyo.Set
            Pyomo Set Object
        """

        # declare_set_with_dict -- If switch off, return empty set
        if not switch:
            return None

        # declare_set_with_dict -- initialize set
        if create_indexed_set:
            sset = pyo.Set(sdata.keys(), initialize=sdata)
        else:
            if use_values:
                sset = pyo.Set(initialize=sdata.values())
            else:
                sset = pyo.Set(initialize=sdata.keys())

        # fix input if single string
        if isinstance(scols, str):
            scols = [scols]

        # check if scols dim is equal to dimension of index
        index_dim = len([[x for x in sdata.keys()][0]])
        if len(scols) != index_dim:
            raise ValueError(
                f"number of index labels provided in scols ({len(scols)}) doesn't match dimension of index \
                in set ({index_dim}). Check set_values and ensure \
                dimension of scols is equal to dimension of elements of values in sdata"
            )

        # declare_set_with_iterable -- update cols_dict with column names
        self.cols_dict[sname] = scols

        # declare_set -- if return not desired, assign to self directly
        if not return_set:
            setattr(self, sname, sset)
            return None
        return sset

    def declare_set_with_sets(
        self,
        sname: str,
        *sets: pyo.Set,
        return_set: bool | None = False,
        switch: bool | None = True,
    ) -> pyo.Set:
        """Declares a new set object using input sets as arguments.

        Function creates a set product with set arguments to create a new set. This is how pyomo
        handles set creation with multiple existing sets as arguments.

        However, this function finds each pyomo set in column dictionary and unpacks the names,
        so that the new set can be logged in the column dictionary too.

        Parameters
        ----------
        sname : str
            Desired name of new set
        *sets : tuple of pyo.Set
            Unnamed arguments assumed to be pyomo sets
        return_set : bool | None, optional
            Return the set rather than assign within function, by default False
        switch : bool | None, optional
            Return None if False, by default True

        Returns
        -------
        pyo.Set
            Pyomo Set Object
        """
        # declare_set_with_sets -- If switch off, return empty set
        if not switch:
            return None

        # declare_set_with_sets -- Unpack arguments, create new name column, and new set product
        new_set = self.unpack_set_arguments(sname, sets)

        # declare_set_with_sets -- unpack and create new set
        sset = pyo.Set(initialize=new_set)
        if not return_set:
            setattr(self, sname, sset)
            return None
        return sset

    def declare_ordered_time_set(self, sname: str, *sets: pyo.Set, return_set: bool | None = False):
        """Unnest the time sets into a single, unnested ordered, synchronous time set, an IndexedSet
        object keyed by the values in the time set, and an IndexedSet object keyed by the combined,
        original input sets.

        These three set outputs are directly set as attributes of the model instance:

        sname:               (1,) , (2, ), ... ,(N)
        sname_time_to_index: (1,):[set1, set2, set3] , (2,):[set1, set2, set3]
        sname_index_to_time: (set1, set2, set3): [1] , (set1, set2, set3): [2]

        In summary, this function creates three sets, creating a unique, ordered set from input sets
        with the assumption that they are given to the function in hierarchical order. For example,
        for a desired time set that orders Year, Month, Day values, the args for the function
        should be provided as:

        m.Year, m.Month, m.Day

        Pyomo set products are used to unpack and create new set values that are ordered by the
        hierarchy provided:

        (year1, month1, day1) , (year1, month1, day2) , ... , (year2, month1, day1) , ... (yearY, monthM, dayD)


        Parameters
        ----------
        sname : str
            Desired root name for the new sets
        sets : pyo.Set
            A series of unnamed arguments assumed to contain pyo.Set in order of temporal hierarchy
        return_set : bool | None, optional
            Return the set rather than assign within function, by default False

        Returns
        -------
        None
            No return object; all sets assigned to model internally

        Raises
        ------
        ValueError
            "No sets provided in args; provide pyo.Set objects to use this function"
        """
        # declare_ordered_time_set -- Unpack arguments, create new name column, and new set product
        new_set = self.unpack_set_arguments(sname, sets)

        # declare_ordered_time_set -- s
        n_times = len(new_set)
        time = pyo.RangeSet(n_times)

        # declare_ordered_time_set -- create IndexSets to switch between singleton and full tuple
        time_to_index = defaultdict()
        index_to_time = defaultdict()

        for element in time:
            time_to_index[element] = new_set.at(element)
            index_to_time[new_set.at(element)] = [element]

        # declare_ordered_time_sets -- create IndexedSets
        time_to_index_set = pyo.Set(time, initialize=time_to_index)
        index_to_time_set = pyo.Set(index_to_time.keys(), initialize=index_to_time)

        # declare_ordered_time_sets -- assign sets
        time_setname = f'{sname}'
        time_index_setname = f'{sname}_time_to_index'
        index_time_setname = f'{sname}_index_to_time'

        setattr(self, time_setname, time)
        setattr(self, time_index_setname, time_to_index_set)
        setattr(self, index_time_setname, index_to_time_set)

    # I'd love for someone to rewrite this; need to move on for now
    def declare_shifted_time_set(
        self,
        sname: str,
        shift_size: int,
        shift_type: Literal['lag', 'lead'],
        *sets: pyo.Set,
        return_set: bool | None = False,
        shift_sets: List | None = None,
    ):
        """A generalize shifting function that creates sets compatible with leads or lags in pyomo
        components.

        For example, with a storage constraint where the current value is contrained to be equal to
        the value of storage in the previous period:

        model.storage[t] == model.storage[t-1] + ...

        The indexing set must be consistent with the storage variable, but not include elements that
        are undefined for this constraint. In this example, the set containing values for t must not
        include t = 1 (e.g. the lagged value must be defined). This function creates a shifted time
        set by removing values from the input sets to comply with the lags or leads.

        Function inputs require a shift size (in the example above, this would be 1), a shift type
        (lead or lag), and the sets used to construct the new, shifted set (model.timestep). If a
        lag or lead is required on a single dimension of the new set, the 'shift_sets' argument can
        include a list of pyo.set names (included in the arguments) to shift by the other args.

        For example...

        model.storage[hub, season] == model.storage[hub, season - 1]

        In this case, season = 1 is always invalid due to the lag; so index (1, 2) or the value for
        hub = 1 and season = 2 is valid, but (2, 1) remains an invalid argument as there is no
        season = 0. A new set composed of hub and season, with shift_sets = ["season"] and
        sets = model.hub, model.season, is created to lag on one index value while leaving others
        unchanged.

        Default is to create set product of all input sets and lag/lead w/ resulting elements.

        Parameters
        ----------
        sname : str
            Desired name for new set
        shift_size : int
            Size of shift in set
        shift_type : str in ["lag", "lead"]
            Type of shift (e.g. t-1 or t+1)
        *sets: Unnamed arguments
            A series of unnamed arguments assumed to contain pyo.Set in order of temporal hierarchy
        return_set : bool | None, optional
            Return the set rather than assign within function, by default False
        shift_sets : List | None, optional
            List of pyo.Set (by name) in *sets to shift, by default None

        Returns
        -------
        pyo.Set
            A pyomo Set

        Raises
        ------
        ValueError
            Shift sets don't align with *sets names
        ValueError
            Type argument is neither lead nor lag
        """
        # declare_ordered_time_set  -- Unpack sets and assign indices to column dictionary; no need
        # to return set product
        self.unpack_set_arguments(sname, sets, return_set_product=False)

        # declare_lagged_time_set  -- if time_sets provided, use timesets to lag; unpack into list
        # to modify set objects
        setnames = [x.getname() for x in sets]
        sets = [x for x in sets]
        if not shift_sets:
            shift_sets = [setnames[-1]]

        # declare_lagged_time_set -- remove necessary elements for shifted sets
        for shift_set in shift_sets:
            if shift_set not in setnames:
                raise ValueError(
                    f'You provided a shift set that was not included in the set arguments: {shift_set}'
                )

            shift_set_index = setnames.index(shift_set)
            shifted_set = [x for x in sets[shift_set_index]]
            match shift_type:
                case 'lag':
                    shifted_set = shifted_set[shift_size:]
                case 'lead':
                    shifted_set = shifted_set[:-shift_size]
                case _:
                    raise ValueError('type argument was neither lead nor lag')
            sets[shift_set_index] = pyo.Set(initialize=shifted_set)

        # declared_lagged_time_set -- create new set product
        n_sets = len(sets)
        new_set = sets[0]
        if n_sets > 1:
            for sset in sets[1:]:
                new_set = new_set * sset

        # declare_lagged_time_set -- create new set and return if desired
        if not return_set:
            setattr(self, sname, new_set)
            return None
        return sset

    def declare_param(
        self,
        pname: str,
        p_set: pyo.Set | None,
        data: dict | pd.DataFrame | pd.Series | int | float,
        return_param: bool | None = False,
        default: int | None = 0,
        mutable: bool | None = False,
    ) -> pyo.Param:
        """Declares a pyo Parameter component named 'pname' with the input data and index set.

        Unpacks column dictionary of index set for param instance and creates pyo.Param; either
        assigns the value internally or returns the object based on return_param.

        Parameters
        ----------
        pname : str
            Desired name of new pyo.Param instance
        p_set : pyo.Set
            Pyomo Set instance to index new Param
        data : dict | pd.DataFrame | pd.Series
            Data to initialize Param instance
        return_param : bool | None, optional
            Return the param after function call rather than assign to self, by default False
        default : int | None, optional
            pyo.Param keyword argument, by default 0
        mutable : bool | None, optional
            pyo.Param keyword argument, by default False

        Returns
        -------
        pyo.Param
            A pyomo Parameter instance

        Raises
        ------
        ValueError
            Raises error if input data not in format supported by function
        """
        # declare_param -- For pandas objects, create param
        if isinstance(data, (pd.DataFrame, pd.Series)):
            # create param
            param = pyo.Param(p_set, initialize=data, default=default, mutable=mutable)

            # update cols_dict
            pcols = list(data.reset_index().columns)
            pcols = pcols[-1:] + pcols[:-1]

        # declare_param -- for dictionaries, create param
        elif isinstance(data, (dict, defaultdict)):
            # create param
            param = pyo.Param(p_set, initialize=data, default=default, mutable=mutable)

            # Get set names from cols_dict and update with new entry
            pcols = self.cols_dict[p_set.getname()]
            pcols = [pname, pcols]

        # declare_param -- for integers and floats
        elif isinstance(data, (int, float)):
            param = pyo.Param(initialize=data, default=default, mutable=mutable)
            pcols = [pname, 'None']

        # declare_param -- throw error if datatype not supported
        else:
            raise ValueError(f'Data type {type(data)} currently unsupported in declare_set method')

        # declare_param -- assign cols to cols_dict
        self.cols_dict[pname] = pcols

        # declare_param -- If return, return the param object or set attribute and return None
        if not return_param:
            setattr(self, pname, param)
            return None
        return param

    def declare_var(
        self,
        vname: str,
        v_set: pyo.Set,
        return_var: bool | None = False,
        within: Literal['NonNegativeReals', 'Binary', 'Reals', 'NonNegativeIntegers']
        | None = 'NonNegativeReals',
        bound: tuple | None = (0, 1000000000),
        switch: bool | None = True,
    ) -> pyo.Var:
        """Declares a pyo Variable component named 'vname' with index set 'v_set'.

        Creates variable indexed by previously defined pyo Set instance 'v_set' and assigns to self;
        function will return the component if return_var is set to True. Other keywords passed to
        pyo.Var are within and bound.

        Parameters
        ----------
        vname : str
            Desired name of new pyo Variable
        v_set : pyo.Set
            Index set for new pyo Variable
        return_var : bool | None, optional
            Return component rather than assign internally, by default False
        within : str in ["NonNegativeReals", "Binary", "Reals", "NonNegativeIntegers"] | None, optional
            pyo.Var keyword argument, by default "NonNegativeReals"
        bound : tuple | None, optional
            pyo.Var keyword argument, by default (0, 1000000000)

        Returns
        -------
        pyo.Var
            A pyomo Variable instance
        """
        # declare_var -- If switch off, return empty set
        if not switch:
            return None

        # declare_var -- Get domain object
        domain = getattr(pyo, within)

        # declare_var -- Create pyo.Var object with domain and bound
        var = pyo.Var(v_set, within=domain, bounds=bound)

        # declare_var -- Update cols_dict with variable name and index column names
        sname = v_set.getname()
        vcols = [vname] + self.cols_dict[sname]
        self.cols_dict[vname] = vcols

        # declare_var -- If not returning the variable, assign to model or return the var object
        if not return_var:
            setattr(self, vname, var)
            return None
        return var

    def unpack_set_arguments(
        self, sname: str, sets: Tuple[pyo.Set], return_set_product: bool | None = True
    ) -> pyo.Set:
        """Handles unnamed pyo.Set arguments for multiple declaration functions.

        For an arbitrarily large number of set inputs, this function unpacks the names for each set
        stored in the column dictionary, creates a new list of the index labels and ordering, and
        then provides the pyo.Set product result as an output.

        Parameters
        ----------
        sname: str
            Name of new set
        sets: tuple of pyo.Set
            Tuple of pyo.Set arguments to be used to generate new set
        return_set_product: bool
            If True, return the unpacked set product

        Returns
        -------
        new_set : pyo.Set
            Set product result from input sets, by order of sets arguments

        """
        # declare_ordered_time_set  -- Check for args
        n_sets = len(sets)
        did_you_provide_any_sets = n_sets > 0
        if not did_you_provide_any_sets:
            raise ValueError(
                'No sets provided in *args; provide pyo.Set objects to use this function'
            )

        # declare_ordered_time_set -- get index names for index_to_time set
        scols = []
        for sset in sets:
            scols = scols + self.cols_dict[sset.getname()]
        self.cols_dict[sname] = scols

        # declare_ordered_time_set  -- unpack sets and save index names to cols_dict
        if return_set_product:
            new_set = sets[0]
            if n_sets > 1:
                for sset in sets[1:]:
                    new_set = new_set * sset
            return new_set
        else:
            return None

    def populate_sets_rule(m1, sname, set_base_name=None, set_base2=None) -> pyo.Set:
        """Generic function to create a new re-indexed set for a pyomo ConcreteModel instance which
        should speed up build time. Must pass non-empty (either) set_base_name or set_base2

        Parameters
        ----------
        m1 : pyo.ConcreteModel
            pyomo model instance
        sname : str
            name of input pyomo set to base reindexing
        set_base_name : str, optional
            the name of the set to be the base of the reindexing, if left blank, uses set_base2, by default ''
        set_base2 : list, optional
            the list of names of set columns to be the base of the reindexing, if left blank, should
            use set_base_name, by default [] these will form the index set of the indexed set structure

        Returns
        -------
        pyomo set
            reindexed set to be added to model
        """

        # get a hook to the input set
        set_in = getattr(m1, sname)
        # get a hook to the column names for the set...
        scols = m1.cols_dict[sname]
        # organize/locate the names of the index set based on input of set_base_name or set_base2
        if set_base_name and set_base2:
            raise ValueError(
                f'Populate function is either-or for set_base_name and set_base2, received both: {set_base_name}, {set_base2}'
            )
        label_locs = None
        index_labels = set()
        if set_base_name:
            label_locs = scols.index(set_base_name)
            index_labels.add(set_base_name)
        elif set_base2:
            try:
                label_locs = [scols.index(t) for t in set_base2]
            except ValueError:
                missing_elements = set(set_base2) - set(scols)
                raise ValueError(
                    f'These elements from set_base2 are not in the base set: {missing_elements}'
                )
            index_labels.update(set_base2)
            # look for case where we did NOT match all of the desired targets in set_base_2
        else:
            raise ValueError('Neither base name or set_base_2 provided')

        # pick up the "data" elements that are not in the index, in current order
        data_indices = [scols.index(t) for t in scols if t not in index_labels]

        # build dictionary result of the indexed set from the input set
        res = defaultdict(list)
        for element in set_in:
            idx = (
                tuple(element[t] for t in label_locs)
                if isinstance(label_locs, list)
                else element[label_locs]
            )
            data = tuple(element[t] for t in data_indices)
            res[idx].append(data)

        if res:
            # make the indexed set from keys -> index set, data = the dictionary
            indexing_set = pyo.Set(initialize=res.keys())
            return pyo.Set(indexing_set, initialize=res)

        # fall back to empty indexed sets keyed by the requested base indices
        if set_base_name:
            base_sets = [list(getattr(m1, set_base_name))]
        else:
            base_sets = [list(getattr(m1, base_name)) for base_name in set_base2]

        if not base_sets:
            indexing_entries = []
        elif len(base_sets) == 1:
            indexing_entries = base_sets[0]
        else:
            from itertools import product

            indexing_entries = list(product(*base_sets))

        indexing_set = pyo.Set(initialize=indexing_entries)
        empty_initializer = {idx: [] for idx in indexing_entries}
        return pyo.Set(indexing_set, initialize=empty_initializer)

    def get_duals(self, component_name: str) -> defaultdict:
        """Extract duals from a solved model instance

        Parameters
        ----------
        component_name : str
            Name of constraint

        Returns
        -------
        defaultdict
            Dual values w/ index values
        """
        # get_duals: get component
        component = getattr(self, component_name)
        if not isinstance(component, pyo.Constraint):
            raise ValueError(
                f'Component {component} is not a pyo.Constraint object; cannot extract duals'
            )

        # get_duals: set up dictionary to capture duals
        dual_dict = defaultdict()
        for index in component:
            dual_dict[index] = self.dual[component[index]]

        # get_duals: return values
        return dual_dict

    ###
    # Classes for decorator expressions
    ###

    class DefaultDecorator:
        """Default decorator class that handles assignment of model scope/pointer in order to use
        pyomo-style parameter and constraint decorators.

        Upon initialization, the decorator handles model assignment at class level to ensure
        inheriting classes have access to the models within local scope.
        """

        def __init__(self, model, *args, **kwargs):
            self.assign_model(model)

        @classmethod
        def assign_model(cls, model):
            """Class-method that assigns a model instance to DefaultDecorator

            Parameters
            ----------
            model : pyo.ConcreteModel
                A pyo model instance
            """
            cls.model = model

    class ParameterExpression(DefaultDecorator):
        """Parameter Expression decorator that works the same as pyomo decorators, while keeping
        column dictionary updated for any indexed parameters given.

        """

        def __init__(self, *args, **kwargs):
            """Upon initialization, assign the model instance stored in DefaultDecorator to scope of
            constraint expression. Named and unnamed keywords assigned to instance.

            Arguments must be pyomo set instances, and named keywords need to be additional params
            passed to pyo.Constraint. This works in practice the same as decorator use in Pyomo.

            @model.ParameterExpression(model.set1, model.set2, keywordarg = "value")
            """
            self._args = list(args)
            self._kwargs = kwargs
            self.instance = super().model

        def __call__(self, expression):
            """Upon instantiation, __call__ unpacks indices for each argument in decorator and
            assigns to cols_dict. Keyword arguments unpack into creation of parameter object, and
            then parameter is assigned to the model instance pointed to in DefaultDecorator

            Parameters
            ----------
            expression : function
                Decorated function
            """
            # wrapper -- use expression name and set names to update cols_dict
            pname = expression.__name__
            scols = []
            for sset in self._args:
                sname = sset.getname()
                scols = [*scols, *self.instance.cols_dict[sname]]

                # check if indexed_set; if so, replace w/ keys
                if sset.is_indexed():
                    warning = f'Provided ParameterExpression with IndexedSet {sname} as Set argument for {pname}; using keys to create OrderedScalarSet'
                    print(warning)
                    self._args[self._args.index(sset)] = pyo.Set(initialize=sset.keys())

            self.instance.cols_dict[pname] = [pname] + scols

            # wrapper -- declare and create constraint
            param = pyo.Param(*self._args, initialize=expression, **self._kwargs)
            setattr(self.instance, pname, param)
            return expression

    class ConstraintExpression(DefaultDecorator):
        """Constraint Expression decorator that works the same as pyomo decorators, while keeping
        column dictionary updated for any indexed parameters given.
        """

        def __init__(self, *args, **kwargs):
            """Upon initialization, assign the model instance stored in DefaultDecorator to scope of
            constraint expression. Named and unnamed keywords assigned to instance.

            Arguments must be pyomo set instances, and named keywords need to be additional params
            passed to pyo.Constraint. This works in practice the same as decorator use in Pyomo.

            @model.ConstraintExpression(model.set1, model.set2, keywordarg = "value")
            """
            self._args = list(args)
            self._kwargs = kwargs
            self.instance = super().model

        def __call__(self, expression):
            """Upon instantiation, __call__ unpacks indices for each argument in decorator and
            assigns to cols_dict. Keyword arguments unpack into creation of constraint object, and
            then constraint is assigned to the model instance pointed to in DefaultDecorator

            Parameters
            ----------
            expression : function
                Decorated function
            """
            # wrapper -- use expression name and set names to update cols_dict
            cname = expression.__name__
            scols = []
            for sset in self._args:
                sname = sset.getname()
                scols = [*scols, *self.instance.cols_dict[sname]]

                # check if indexed_set; if so, replace w/ keys
                if sset.is_indexed():
                    warning = f'Provided ConstraintExpression with IndexedSet {sname} as Set argument for {cname}; using keys to create OrderedScalarSet'
                    print(warning)
                    self._args[self._args.index(sset)] = pyo.Set(initialize=sset.keys())

            self.instance.cols_dict[cname] = [cname] + scols

            # wrapper -- declare and create constraint
            constraint = pyo.Constraint(*self._args, expr=expression, **self._kwargs)
            setattr(self.instance, cname, constraint)
            return expression
