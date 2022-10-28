# Copyright © 2021 United States Government as represented by the Administrator of the
# National Aeronautics and Space Administration.  All Rights Reserved.

"""
Example demonstrating ways to use the derived parameters feature for model building. 

.. dropdown:: More details
    
    In this example, a derived parameter (i.e., a parameter that is a function of another parameter) are defined for the simple ThrownObject model. These parameters are then calculated whenever their dependency parameters are updated, eliminating the need to calculate each timestep in simulation. The functionality of this feature is then demonstrated.
"""

from prog_models.models.thrown_object import ThrownObject

def run_example():
    # For this example we will use the ThrownObject model from the new_model example.
    # We will extend that model to include a derived parameter
    # Let's assume that the throwing_speed was actually a function of thrower_height 
    # (i.e., a taller thrower would throw the ball faster).
    # Here's how we would implement that

    # Step 1: Define a function for the relationship between thrower_height and throwing_speed.
    def update_thrown_speed(params):
        return {
            'throwing_speed': params['thrower_height'] * 21.85
        }  # Assumes thrown_speed is linear function of height
    # Note: one or more parameters can be changed in these functions, whatever parameters are changed are returned in the dictionary

    # Step 2: Define the param callbacks
    ThrownObject.param_callbacks.update({
            'thrower_height': [update_thrown_speed]
        })  # Tell the derived callbacks feature to call this function when thrower_height changes.
    # Note: Usually we would define this method within the class
    #  for this example, we're doing it separately to improve readability
    # Note2: You can also have more than one function be called when a single parameter is changed.
    #  Do this by adding the additional callbacks to the list (e.g., 'thrower_height': [update_thrown_speed, other_fcn])

    # Step 3: Use!
    obj = ThrownObject()
    print("Default Settings:\n\tthrower_height: {}\n\tthowing_speed: {}".format(obj.parameters['thrower_height'], obj.parameters['throwing_speed']))
    
    # Now let's change the thrower_height
    print("changing height...")
    obj.parameters['thrower_height'] = 1.75  # Our thrower is 1.75 m tall
    print("\nUpdated Settings:\n\tthrower_height: {}\n\tthowing_speed: {}".format(obj.parameters['thrower_height'], obj.parameters['throwing_speed']))
    print("Notice how speed changed automatically with height")


# This allows the module to be executed directly 
if __name__ == '__main__':
    run_example()
