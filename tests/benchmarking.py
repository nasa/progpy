# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.

from io import StringIO
import sys
import timeit
from time import process_time
from progpy.models import ThrownObject

FORMAT_STR = '| {:40s} |'

if __name__ == "__main__":
    print('| Test | Time (s) |\n| --- | --- |')

    # print(FORMAT_STR.format('import main'), end='')
    # t = timeit.timeit('import progpy', timer=process_time)
    # print(f'{t} |')
    #
    # print(FORMAT_STR.format('import thrown object'), end='')
    # t = timeit.timeit('from progpy.models import ThrownObject', timer=process_time)
    # print(f'{t} |')
    #
    # print(FORMAT_STR.format('model initialization'), end='')
    # t = timeit.timeit('ThrownObject()', 'from progpy.models import ThrownObject', number=1000, timer=process_time)
    # print(f'{t} |')
    #
    # m = ThrownObject()
    #
    # print(FORMAT_STR.format('set noise'), end='')
    # t = timeit.timeit("m.parameters['process_dist'] = 'none'", 'from progpy.models import ThrownObject; m = ThrownObject()', timer=process_time)
    # print(f'{t} |')

    # print(FORMAT_STR.format('simulate'), end='')
    # t = timeit.timeit("m.simulate_to_threshold(future_load, threshold_keys='impact')", 'from progpy.models import ThrownObject; m = ThrownObject(); future_load = lambda t, x=None : m.InputContainer([{}])', number=1000, timer=process_time)
    # print(f'{t} |')

    # print(FORMAT_STR.format('simulate with saving'), end='')
    # t = timeit.timeit("m.simulate_to_threshold(future_load, threshold_keys='impact', save_freq = 0.5)", 'from progpy.models import ThrownObject; m = ThrownObject(); future_load = lambda t, x=None : m.InputContainer({})', number=1000, timer=process_time)
    # print(f'{t} |')

    # print(FORMAT_STR.format('simulate with saving, dt'), end='')
    # t = timeit.timeit("m.simulate_to_threshold(future_load, threshold_keys='impact', save_freq = 0.5, dt=0.1)", 'from progpy.models import ThrownObject; m = ThrownObject(); future_load = lambda t, x=None : m.InputContainer({})', number=500, timer=process_time)
    # print(f'{t} |')

    print(FORMAT_STR.format('simulate with printing results, dt'), end='')
    temp_out = StringIO()
    sys.stdout = temp_out
    t = timeit.timeit("m.simulate_to_threshold(future_load, threshold_keys='impact', save_freq = 0.5, dt=0.1, print = True)", 'from progpy.models import ThrownObject; m = ThrownObject(); future_load = lambda t, x=None : m.InputContainer({})', number=500, timer=process_time)
    sys.stdout = sys.__stdout__
    print(f'{t} |')
    #
    # # print(FORMAT_STR.format('Plot results'), end='')
    # # t = timeit.timeit("result.outputs.plot()", "from prog_models.models import ThrownObject; m = ThrownObject(); future_load = lambda t, x=None : m.InputContainer({}); result = m.simulate_to_threshold(future_load, threshold_keys='impact', save_freq = 0.5, dt=0.1)", number=1000, timer=process_time)
    # # print(f'{t} |')
    #
    # print(FORMAT_STR.format('Metrics'), end='')
    # t = timeit.timeit("result.event_states.monotonicity()", "from progpy.models import ThrownObject; m = ThrownObject(); future_load = lambda t, x=None : m.InputContainer({}); result = m.simulate_to_threshold(future_load, threshold_keys='impact', save_freq = 0.5, dt=0.1)", number=1000, timer=process_time)
    # print(f'{t} |')
    #
    # print(FORMAT_STR.format('Surrogate Model Generation'), end='')
    # temp_out = StringIO()
    # sys.stdout = temp_out
    # sys.stderr = temp_out
    # t = timeit.timeit("m.generate_surrogate([future_load], threshold_keys='impact')", "from progpy.models import ThrownObject; m = ThrownObject(); future_load = lambda t, x=None : m.InputContainer({})", number=1000, timer=process_time)
    # sys.stdout = sys.__stdout__
    # sys.stderr = sys.__stderr__
    # print(f'{t} |')
    #
    # print(FORMAT_STR.format('surrogate sim'), end='')
    # temp_out = StringIO()
    # sys.stdout = temp_out
    # sys.stderr = temp_out
    # t = timeit.timeit("m2.simulate_to_threshold(future_load, threshold_keys='impact')", "from progpy.models import ThrownObject; m = ThrownObject(); future_load = lambda t, x=None : m.InputContainer([{}]); m2 = m.generate_surrogate([future_load], threshold_keys='impact')", number=1000, timer=process_time)
    # sys.stdout = sys.__stdout__
    # sys.stderr = sys.__stderr__
    # print(f'{t} |')
    #
    # print(FORMAT_STR.format('surrogate sim, dt'), end='')
    # temp_out = StringIO()
    # sys.stdout = temp_out
    # sys.stderr = temp_out
    # t = timeit.timeit("m2.simulate_to_threshold(future_load, threshold_keys='impact', save_freq=0.25)", "from progpy.models import ThrownObject; m = ThrownObject(); future_load = lambda t, x=None : m.InputContainer({}); m2 = m.generate_surrogate([future_load], threshold_keys='impact')", number=1000, timer=process_time)
    # sys.stdout = sys.__stdout__
    # sys.stderr = sys.__stderr__
    # print(f'{t} |')
