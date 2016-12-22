from copy import copy
import re

class PhaseSpace(object):
    """A PhaseSpace defines the possible combinations of variables that characterize an event.

    It can be seen as the carthesian product of those variables.

        >>> ps = PhaseSpace(variables=['a', 'b', 'c'])
        >>> print ps
        ('a' X 'c' X 'b')

    You can check whether a variable is part of a phase space:

        >>> 'a' in ps
        True

    Phase spaces can be compared to one another.

    Check whether two phase spaces are identical:

        ('a' X 'b') == ('a' X 'b')
        ('a' X 'b') != ('a' X 'c')

    Check whether one phase space is a sub-space of the other:

        ('a' X 'b' X 'c') > ('a' X 'b')
        ('a' X 'c') < ('a' X 'b' X 'c')

    """

    def __init__(self, variables):
        """Create a PhaseSpace object.

        Arguments
        ---------

        variables: The set of variables that define the phase space.
        """

        self.variables = set(variables)

    def __contains__(self, var):
        return var in self.variables

    def __eq__(self, phasespace):
        return self.variables == phasespace.variables

    def __ne__(self, phasespace):
        return not self.variables == phasespace.variables

    def __le__(self, phasespace):
        return self.variables <= phasespace.variables

    def __ge__(self, phasespace):
        return self.variables >= phasespace.variables

    def __lt__(self, phasespace):
        return (self.variables <= phasespace.variables) and not (self.variables == phasespace.variables)

    def __gt__(self, phasespace):
        return (self.variables >= phasespace.variables) and not (self.variables == phasespace.variables)

    def __mul__(self, phasespace):
        return PhaseSpace(variables = (self.variables | phasespace.variables))

    def __div__(self, phasespace):
        return PhaseSpace(variables = (self.variables - phasespace.variables))

    def __str__(self):
        return "('" + "' X '".join(self.variables) + "')"

    @staticmethod
    def from_string(string):
        """Create a PhaseSpace object from it string representation."""
        m = re.match(r"\('(.+)'\)$", string)
        string = m.group(1)
        variables = re.split(r"' X '", string)
        return PhaseSpace(variables=variables)

class Bin(object):
    """A Bin is container for a value that is defined on a subset of an n-dimensional phase space."""

    def __init__(self, **kwargs):
        """Create basic bin.

        kwargs
        ------

        phasespace : The phase space the Bin resides in.
        value : The initialization value of the bin. Default: 0.
        """

        self.phasespace = kwargs.pop('phasespace', None)
        if self.phasespace is None:
            raise ValueError("Undefined phase space!")

        self.value = kwargs.pop('value', 0.)

        if len(kwargs) > 0:
            raise ValueError("Unknown kwargs: %s"%(kwargs,))

    def event_in_bin(self, event):
        """Return True if the variable combination falls within the bin."""

        raise NotImplementedError("This method must be defined in an inheriting class.")

    def fill(self, weight=1.):
        """Add the weight(s) to the bin."""

        try:
            self.value += sum(weight)
        except TypeError:
            self.value += weight

    def __contains__(self, event):
        """Return True if the event falls within the bin."""
        return self.event_in_bin(event)

    def __add__(self, other):
        ret = copy(self)
        ret.value = self.value + other.value
        return ret

    def __sub__(self, other):
        ret = copy(self)
        ret.value = self.value - other.value
        return ret

    def __mul__(self, other):
        ret = copy(self)
        ret.value = self.value * other.value
        return ret

    def __div__(self, other):
        ret = copy(self)
        ret.value = self.value / other.value
        return ret

    def __str__(self):
        return "Bin on phase space %s"%(self.phasespace,)

class RectangularBin(Bin):
    """A bin defined by min and max values in all variables."""

    def __init__(self, **kwargs):
        """Initialize a rectangular bin with bin edges.

        kwargs
        ------

        edges: A dict of {'varname': (lower_edge, upper_edge)}
        include_lower: Does the bin include the lower edges? Default: True
        include_upper: Does the bin include the upper edges? Default: False
        """

        self.include_lower = kwargs.pop('include_lower', True)
        self.include_upper = kwargs.pop('include_upper', False)
        self.edges = kwargs.pop('edges', None)
        if self.edges is None:
            raise ValueError("Edges are not defined")

        # Create PhaseSpace from edges if necessary
        phasespace = kwargs.get('phasespace', None)
        if phasespace is None:
            kwargs['phasespace'] = PhaseSpace(self.edges.keys())

        # Handle default bin initialization
        Bin.__init__(self, **kwargs)

        # Check that all edges are valid
        for var in self.edges:
            if var not in self.phasespace:
                raise ValueError("Variable not part of PhaseSpace: %s"%(var,))
            mi, ma = self.edges[var]

            if ma < mi:
                raise ValueError("Upper edge is smaller than lower edge for variable %s."%(var,))

    def event_in_bin(self, event):
        """Check whether an event is within all bin edges."""

        inside = True

        for var in self.edges:
            mi, ma = self.edges[var]
            val = event[var]
            if self.include_lower:
                if val < mi:
                    inside = False
                    break
            else:
                if val <= mi:
                    inside = False
                    break
            if self.include_upper:
                if val > ma:
                    inside = False
                    break
            else:
                if val >= ma:
                    inside = False
                    break

        return inside

    def get_center(self):
        """Return the bin center coordinates."""
        center = {}
        for key, (mi, ma) in self.edges.items():
            center[key] = (float(mi) + float(ma)) / 2.
        return center

class Binning(object):
    """A Binning is a set of Bins.

    It translates variable values to bin numbers and vice versa.
    """

    def __init__(self, **kwargs):
        """Create basic Binning.

        kwargs
        ------

        phasespace : The PhaseSpace the Binning resides in.
        bins : The list of disjoint bins on that PhaseSpace.
        """

        self.phasespace = kwargs.pop('phasespace', None)
        if self.phasespace is None:
            raise ValueError("Undefined phase space!")

        self.bins = kwargs.pop('bins', None)
        if self.bins is None:
            raise ValueError("Undefined bins!")

        # Check that all bins are defined on the given phase space
        for b in self.bins:
            if b.phasespace != self.phasespace:
                raise ValueError("Phase space of bin does not match phase space of binning!")

        if len(kwargs) > 0:
            raise ValueError("Unknown kwargs: %s"%(kwargs,))

    def get_event_bin_number(self, event):
        """Returns the bin number of the given event.

        Returns `None` if the event does not belong to any bin.

        This is a dumb method that just loops over all bins until it finds a fitting one.
        It should be replaced with something smarter for more specifig binning classes.
        """

        for i in range(len(self.bins)):
            if event in self.bins[i]:
                return i

        return None

    def get_event_bin(self, event):
        """Return the bin of the event.

        Returns `None` if the event does not fit in any bin.
        """

        nr = self.get_event_bin_number(event)
        if nr is not None:
            return self.bins[nr]
        else:
            return None
