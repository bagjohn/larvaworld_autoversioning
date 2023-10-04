import param
from ..param import Named

__all__ = [
    'Object',
    'NamedObject',
    'GroupedObject',
]

__displayname__ = 'Basic ABM class'


class Object:
    """
    Basic Class for all Larvaworld model objects.

    This class extends the agentpy Object class by allowing the recording of nested attributes.

    Parameters
    ----------
    model : object, optional
        The model this object belongs to.
    id : str, optional
        The unique identifier for this object.

    Attributes
    ----------
    id : str
        The unique identifier for this object.
    type : str
        The name of the object's class.
    log : dict
        A dictionary for recording log data.
    model : object
        The model this object belongs to.
    p : object
        The parameters of the model.

    Methods
    -------
    __repr__()
        Return a string representation of the object.
    __getattr__(key)
        Raise an AttributeError for unknown attributes.
    __getitem__(key)
        Get an attribute value.
    __setitem__(key, value)
        Set an attribute value.
    _set_var_ignore()
        Store current attributes to separate them from custom variables.
    vars
        Get a list of attribute names.
    _log
        Access the log data.
    extend_log(l, k, N, v)
        Extend log data with a new value.
    connect_log(ls)
        Connect the log to the model's dictionary of logs.
    nest_record(reporter_dic)
        Record nested attributes.
    setup(**kwargs)
        Initialize object attributes and actions.

    Examples
    --------
    The following setup initializes an object with three variables:

    def setup(self, y):
        self.x = 0  # Value defined locally
        self.y = y  # Value defined in kwargs
        self.z = self.p.z  # Value defined in parameters
    """

    def __init__(self, model=None, id='Nada'):
        self._var_ignore = []
        self.id = id
        self.type = type(self).__name__
        self.log = {}
        self.model = model
        if model is not None:
            self.p = model.p

    def __repr__(self):
        return f"{self.type} (Obj {self.id})"

    def __getattr__(self, key):
        raise AttributeError(f"{self} has no attribute '{key}'.")

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def _set_var_ignore(self):
        """Store current attributes to separate them from custom variables"""
        self._var_ignore = [k for k in self.__dict__.keys() if k[0] != '_']

    @property
    def vars(self):
        return [k for k in self.__dict__.keys()
                if k[0] != '_'
                and k not in self._var_ignore]

    '''
        def record(self, var_keys, value=None):
        """ Records an object's variables at the current time-step.
        Recorded variables can be accessed via the object's `log` attribute
        and will be saved to the model's output at the end of a simulation.

        Arguments:
            var_keys (str or list of str):
                Names of the variables to be recorded.
            value (optional): Value to be recorded.
                The same value will be used for all `var_keys`.
                If none is given, the values of object attributes
                with the same name as each var_key will be used.

        Notes:
            Recording mutable objects like lists can lead to wrong results
            if the object's content will be changed during the simulation.
            Make a copy of the list or record each list entry seperately.

        Examples:

            Record the existing attributes `x` and `y` of an object `a`::

                a.record(['x', 'y'])

            Record a variable `z` with the value `1` for an object `a`::

                a.record('z', 1)

            Record all variables of an object::

                a.record(a.vars)
        """

        # Initial record call

        # Connect log to the model's dict of logs
        if self.type not in self.model._logs:
            self.model._logs[self.type] = {}
        self.model._logs[self.type][self.id] = self.log
        self.log['t'] = [self.model.t]  # Initiate time dimension

        # Perform initial recording
        for var_key in make_list(var_keys):
            v = getattr(self, var_key) if value is None else value
            self.log[var_key] = [v]

        # Set default recording function from now on
        self.record = self._record  # noqa

    def _record(self, var_keys, value=None):

        for var_key in make_list(var_keys):

            # Create empty lists
            if var_key not in self.log:
                self.log[var_key] = [None] * len(self.log['t'])

            if self.model.t != self.log['t'][-1]:

                # Create empty slot for new documented time step
                for v in self.log.values():
                    v.append(None)

                # Store time step
                self.log['t'][-1] = self.model.t

            if value is None:
                v = getattr(self, var_key)
            else:
                v = value

            self.log[var_key][-1] = v
    '''


    def setup(self, **kwargs):
        """This empty method is called automatically at the objects' creation.
        Can be overwritten in custom sub-classes
        to define initial attributes and actions.

        Arguments:
            **kwargs: Keyword arguments that have been passed to
                :class:`Agent` or :func:`Model.add_agents`.
                If the original setup method is used,
                they will be set as attributes of the object.

        Examples:
            The following setup initializes an object with three variables::

                def setup(self, y):
                    self.x = 0  # Value defined locally
                    self.y = y  # Value defined in kwargs
                    self.z = self.p.z  # Value defined in parameters
        """

        for k, v in kwargs.items():
            setattr(self, k, v)


class NamedObject(Named, Object):
    """
    A named simulation object that extends the basic Object class.

    Parameters
    ----------
    model : object, optional
        The model this object belongs to.
    **kwargs
        Additional keyword arguments.

    Attributes
    ----------
    Inherits attributes from Object class.
    """

    def __init__(self, model=None, **kwargs):
        Named.__init__(self, **kwargs)
        Object.__init__(self, model=model, id=self.unique_id)


class GroupedObject(NamedObject):
    """
    A grouped simulation object that extends the NamedObject class.

    Attributes
    ----------
    group : str, optional
        The unique ID of the entity's group.

    Methods
    -------
    Inherits methods from NamedObject class.
    """

    group = param.String(None, doc="The unique ID of the entity's group")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.group is not None:
            self.type = self.group
