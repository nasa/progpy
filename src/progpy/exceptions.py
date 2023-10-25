# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

from warnings import warn


class ProgModelStateLimitWarning(Warning):
    """
    Prognostics State Limit Warning - indicates the model state was outside the limits, and was adjusted
    """


warnings_seen = set()


def warn_once(msg, *args, **kwargs):
    if msg not in warnings_seen:
        # First time warning
        warnings_seen.add(msg)
        warn(msg, *args, **kwargs)
