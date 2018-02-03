
Dmipy Contribution Guidelines
=============================

The idea behind dmipy is that it's modular and easily extendable with
new models and optimizers.

To contribute your work to dmipy, we just ask that you adhere to some
guidelines as to how to structure the functions. You can contribute
CompartmentModel, Spherical Distributions, Spatial Distributions and
Optimizers. Their blueprints are given below.

If you want contribute, just contact us or open a pull request.

Blueprint Compartment Models
----------------------------

.. code:: ipython2

    class NewCompartmentModel:
        
        # some default optimization ranges (min and max allowed value), and their scale.
        self._parameter_scales = {'parameter1': value1, 'parameter2': value2}
        self._parameter_ranges = {'parameter1': [start1, end1], 'parameter2': [start2, end2]}
        
        def __init__(self, parameter1=None, parameter2=None):
            "instantiate your model"
            self.parameter1 = parameter1
            self.parameter2 = parameter2
        
        def __call__(self, acquisition_scheme, **kwargs):
            "your function call that returns signal attenuation for the given acquisition scheme"
            
            parameter1 = kwargs.get('parameter1', self.parameter1)
            parameter2 = kwargs.get('parameter2', self.parameter2)
    
            return signal_attenuation
    
        def optional_helper_functions(self, params)
            ...

Blueprint Spherical Distributions
---------------------------------

.. code:: ipython2

    class NewSphericalDistribution:
        
        # some default optimization ranges (min and max allowed value), and their scale.
        self._parameter_scales = {'parameter1': value1, 'parameter2': value2}
        self._parameter_ranges = {'parameter1': [start1, end1], 'parameter2': [start2, end2]}
        
        def __init__(self, parameter1=None, parameter2=None):
            "instantiate your model"
            self.parameter1 = parameter1
            self.parameter2 = parameter2
        
        def __call__(self, sphere_orientations, **kwargs):
            "your function call that returns probability density at the given sphere orientations"
            
            parameter1 = kwargs.get('parameter1', self.parameter1)
            parameter2 = kwargs.get('parameter2', self.parameter2)
    
            return probability_density
    
        def spherical_harmonics_representation(self, sh_order=some_default, **kwargs):
            "returns the spherical harmonics representation of the spherical distribution"
            return distribution_sh_coefficients
            
        def optional_helper_functions(self, params)
            ...

Blueprint Spatial Distributions
-------------------------------

.. code:: ipython2

    class NewSpatialDistribution:
        
        # some default optimization ranges (min and max allowed value), and their scale.
        self._parameter_scales = {'parameter1': value1, 'parameter2': value2}
        self._parameter_ranges = {'parameter1': [start1, end1], 'parameter2': [start2, end2]}
        
        def __init__(self, parameter1=None, parameter2=None):
            "instantiate your model"
            self.parameter1 = parameter1
            self.parameter2 = parameter2
        
        def __call__(self, **kwargs):
            "your function call that returns probability density for some sampling range."
            "Ideally this sampling range is automatically dependent on the input parameters."
            
            parameter1 = kwargs.get('parameter1', self.parameter1)
            parameter2 = kwargs.get('parameter2', self.parameter2)
    
            return sampled_parameter_points, probability_density_at_those_points
            
        def optional_helper_functions(self, params)
            ...

Blueprint Optimizers
--------------------

.. code:: ipython2

    class NewOptimizer:
        # the optimizer should be instantiated using the model, acquisition scheme and possible solver options.
        def __init__(self, model, acquisition_scheme, possible_solver_options):
            self.model = model
            self.acquisition_scheme = acquisition_scheme
            self.possible_solver_options = possible_solver_options
        
        def __call__(self, data, possible_x0_vector):
            "function call that returns the fitted model parameters."
            return fitted_parameter_array
        
        def optional_helper_functions(self, params)
            ...
