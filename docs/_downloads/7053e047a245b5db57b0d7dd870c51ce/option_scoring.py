# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
This example demonstrates how to score multiple considered options using the PaaS Sandbox. Prior to running the example start the server in a terminal window with the command:
    python -m prog_server

This example creates a session with the server to run prognostics for a BatteryCircuit. Three options with different loading profiles are compared by creating a session for each option and comparing the resulting prediction metrics
"""

import prog_client
import time

def run_example():
    # Step 1: Prepare load profiles to compare
    # Create a load profile for each option
    # Each load profile has format Array[Dict]
    # Where each dict is in format {TIME: LOAD}
    # The TIME is the start of that loading in seconds
    # LOAD is a dict with keys corresponding to model.inputs
    # Note: Dict must be in order of increasing time
    LOAD_PROFILES = [
        { # Plan 0
            0: {'i': 2},
            600: {'i': 1},
            900: {'i': 4},
            1800: {'i': 2},
            3000: {'i': 3}
        },
        { # Plan 1
            0: {'i': 3},
            900: {'i': 2},
            1000: {'i': 3.5},
            2000: {'i': 2.5},
            2300: {'i': 3}
        },
        { # Plan 2
            0: {'i': 1.25},
            800: {'i': 2},
            1100: {'i': 2.5},
            2200: {'i': 6},
        }
    ]
    
    # Step 2: Open a session with the server for a thrown object. 
    # We are specifying a time of interest of 2000 seconds.
    # This could be the end of a mission/session, or some inspection time. 
    print('\nStarting Sessions')
    sessions = [prog_client.Session('BatteryCircuit', pred_cfg = {'save_pts': [2000], 'save_freq': 1e99, 'n_samples':15}, load_est = 'Variable', load_est_cfg = LOAD_PROFILES[i]) for i in range(len(LOAD_PROFILES))]

    # Step 3: Wait for prognostics to complete
    print('\nWaiting for sessions to complete (this may take a bit)')
    STEP = 15  # Time to wait between pinging server (s)
    
    for session in sessions:
        sessions_in_progress = True
        while sessions_in_progress:
            sessions_in_progress = False
            status = session.get_prediction_status()
            if status['in progress'] != 0:
                print(f'\tSession {session.session_id} is still in progress')
                sessions_in_progress = True
                time.sleep(STEP)
        print(f'\tSession {session.session_id} complete')
    print('All sessions complete')
    
    # Step 4: Get the results
    print('Getting results')
    results = [session.get_predicted_toe()[1] for session in sessions]

    # Step 5: Compare results
    print('\nComparing results')
    print('Mean ToE:')
    best_toe = 0
    best_plan = None
    for i in range(len(results)):
        mean_toe = results[i].mean['EOD']
        print(f'\tOption {i}: {mean_toe:0.2f}s')
        if mean_toe > best_toe:
            best_toe = mean_toe
            best_plan = i
    print(f'Best option using method 1: Option {best_plan}')

    print('\nSOC at point of interest (2000 sec):')
    best_soc = 0
    best_plan = None
    soc = [session.get_predicted_event_state()[1] for session in sessions]
    for i in range(len(soc)):
        mean_soc = soc[i].snapshot(-1).mean['EOD']
        print(f'\tOption {i}: {mean_soc:0.3f} SOC')
        if mean_soc > best_soc:
            best_soc = mean_soc
            best_plan = i
    print(f'Best option using method 2: Option {best_plan}')

    # Other metrics can be used as well, like probability of mission success given a certain mission time, uncertainty in ToE estimate, final state at end of mission, etc. 

# This allows the module to be executed directly
if __name__ == '__main__':
    run_example()
