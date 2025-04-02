# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

import enum
import random

# Transition Functions
def _random_transition(state, disruption):
    """
    Random transition from one state to another if disruption is >= 0.5
    """
    if abs(disruption) >= 0.5:
        return random.randint(0, len(type(state))-1)
    return state

def _sequential_transition(state, disruption):
    """
    Sequential transition from one state to the next if disruption is >= 0.5
    
    Examples:
        Mode 1 + 0.5 -> Mode 2
        Mode 2 - 0.5 -> Mode 1
        Mode 1 + 1.5 -> Mode 3
        Mode 1 + 0.4 -> Mode 1
        Mode 0 - 2 -> Mode 0 (Minimum limit)
    """
    return min(max(state._value_ + round(disruption), 0), len(type(state))-1)

def _no_transition(state, disruption):
    """
    No transition at all. This is used for case where all transitions are done manually (e.g., in state transition equation). This way state will not be affected by noise
    """
    return state

TRANSITION_LOOKUP = {
    'random': _random_transition,
    'sequential': _sequential_transition,
    'none': _no_transition
}

class DiscreteState(enum.Enum):
    """
    .. versionadded:: 1.8.0

    Class for discrete state. Users wont be constructing this directly, but will instead be using the factory function create_discrete_state.

    This is an enum, so discrete states will be accessed directly. For example:
        DiscreteState.state_name
        DiscreteState(1)

    The add method is overwritten to provide logic for transition (according to provided function)
    """
    def __add__(self, other):
        return type(self)(self._transition(other))

    def __sub__(self, other):
        return type(self)(self._transition(-other))

def create_discrete_state(n_states: int, names: list=None, transition=_random_transition) -> DiscreteState:
    """
    .. versionadded:: 1.8.0

    Create a discrete state for use with a progpy model. Users construct a discrete state for the default x0 to make that state discrete.

    Args:
        n_states (int): Number of possible states.
        names (list[str], optional): Names for states. Defaults to using "State [#]" for each state (e.g., "State 1")
        transition ({function, str}, optional): Transition logic. Can be either a string ('random', 'none', or 'sequential') or a function (DiscreteState, float)->int of state and disruption to state number. Defaults to "random".

    Returns:
        DiscreteState class: Class to construct a discrete state

    Example:
        >>> Switch = create_discrete_state(2, ['on', 'off'])
        >>> x0['switch'] = Switch.off

    Example:
        >>> # Representing 'gear' of car
        >>> Gear = create_discrete_state(5, transition='sequential')
        >>> x0['gear] = Gear(1)
    """
    # Input Validation
    if isinstance(transition, str):
        if transition in TRANSITION_LOOKUP:
            transition = TRANSITION_LOOKUP[transition]
        else:
            raise Exception(f'Transition name {transition} not recognized. Supported modes are {str(list(TRANSITION_LOOKUP.keys()))}')

    if names is None:
        names = [f'State {i}' for i in range(n_states)]
    elif len(names) != n_states:
        raise ValueError(f'If providing names, must provide one for each state. Provided {len(names)} for {n_states} states.')

    # Enumerated states
    members = {name: i for i, name in enumerate(names)}

    discrete_state = DiscreteState('Discrete State', members)

    # Transition is set to be nonmember (meaning it's not an enumerated state)
    discrete_state._transition = transition
    
    return discrete_state
