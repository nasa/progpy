# Copyright © 2021 United States Government as represented by the Administrator of the National Aeronautics and Space Administration.  All Rights Reserved.
"""
This example demonstrates a use case where someone wants to predict the first event (i.e., End Of Life (EOL)) of a system. Many system models have multiple events that can occur. In some prognostics applications, users are not interested in predicting a specific event, and are instead interested in when the first event occurs, regardless of the event. This example demonstrates how to predict the first event of a system.

Method: An instance of ThrownObject is used for this example. In this case it is trivial because the event 'falling' will always occur before 'impact', but for some other models that might not be true. The ThrownObject class is subclassed to add a new event 'EOL' which occurs if any other event occurs. The new model is then instantiated and used for prognostics like in basic_example. Prediction specifically specifies EOL as the event to be predicted.

Results: 

    i) Predicted future values (inputs, states, outputs, event_states) with uncertainty from prediction
    ii) Time the event 'EOL' is predicted to occur (with uncertainty)
    iii) Histogram of the event 'EOL'
"""

import matplotlib.pyplot as plt
from prog_models.models import ThrownObject
from prog_algs.predictors import MonteCarlo
from prog_algs.uncertain_data import ScalarData

def run_example():
    # Step 1: Define subclass with EOL event 
    # Similar to the prog_models 'events' example, but with an EOL event
    class ThrownObjectWithEOL(ThrownObject):
        events = ThrownObject.events + ['EOL']

        def event_state(self, x):
            es = super().event_state(x)
            # Add EOL Event (minimum event state)
            es['EOL'] = min(list(es.values()))
            return es
        
        def threshold_met(self, x):
            t_met = super().threshold_met(x)
            # Add EOL Event (if any events have occured)
            t_met['EOL'] = any(list(t_met.values()))
            return t_met
    
    # Step 2: Create instance of subclass
    m = ThrownObjectWithEOL(process_noise=1)

    # Step 3: Setup for prediction
    pred = MonteCarlo(m)
    def future_loading(t=None, x=None):
        return {}  # No future loading for ThrownObject
    state = ScalarData(m.initialize())

    # Step 4: Predict to EOL event
    pred_results = pred.predict(state, future_loading, events=['EOL'], dt=0.01, n_samples=50)
    # In this case EOL is when the object starts falling
    # But for some models where events aren't sequential, there might be a mixture of events in the EOL

    # Step 5: Plot results
    pred_results.time_of_event.plot_hist()
    plt.show()

# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
