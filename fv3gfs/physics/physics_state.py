from dataclasses import dataclass, field, fields, InitVar
from fv3core.utils.typing import FloatField, FloatFieldIJ
import copy
import fv3gfs.util as fv3util 
from fv3gfs.physics.stencils.microphysics import MicrophysicsState


@dataclass()
class PhysicsState:
    qvapor: FloatField = field(metadata={"name": "specific_humidity", "units": "kg/kg", "from_dycore": True})
    qliquid: FloatField = field(metadata={"name": "cloud_water_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": True})
    qice: FloatField = field(metadata={"name": "cloud_ice_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": True}) 
    qrain: FloatField = field(metadata={"name": "rain_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": True})
    qsnow: FloatField = field(metadata={"name": "snow_mixing_ratio","units": "kg/kg", "intent":"inout", "from_dycore": True})
    qgraupel: FloatField = field(metadata={"name": "graupel_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": True}) 
    qo3mr: FloatField = field(metadata={"name": "ozone_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": True}) 
    qsgs_tke: FloatField = field(metadata={"name": "turbulent_kinetic_energy","units": "m**2/s**2", "intent":"inout", "from_dycore": True})
    qcld: FloatField = field(metadata={"name": "cloud_fraction","units": "", "intent":"inout", "from_dycore": True})
    pt: FloatField = field(metadata={"name": "air_temperature", "units": "degK", "intent":"inout", "from_dycore": True})
    delp: FloatField = field(metadata={"name": "pressure_thickness_of_atmospheric_layer", "units": "Pa", "intent":"inout", "from_dycore": True})
    delz: FloatField = field(metadata={"name": "vertical_thickness_of_atmospheric_layer", "units": "m", "intent":"inout", "from_dycore": True}) 
    ua: FloatField = field(metadata={"name": "eastward_wind", "units": "m/s", "intent":"inout", "from_dycore": True})
    va: FloatField = field(metadata={"name": "northward_wind", "units": "m/s", "from_dycore": True })
    w: FloatField = field(metadata={"name": "vertical_wind", "units": "m/s", "intent":"inout", "from_dycore": True})
    omga: FloatField = field(metadata={"name": "vertical_pressure_velocity","units": "Pa/s", "intent":"inout", "from_dycore": True})
    qvapor_t1: FloatField = field(metadata={"name": "physics_specific_humidity", "units": "kg/kg", "from_dycore": False})
    qliquid_t1: FloatField = field(metadata={"name": "physics_cloud_water_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": False})
    qice_t1: FloatField  = field(metadata={"name": "physics_cloud_ice_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": False}) 
    qrain_t1: FloatField =  field(metadata={"name": "physics_rain_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": False})
    qsnow_t1: FloatField = field(metadata={"name": "physics_snow_mixing_ratio","units": "kg/kg", "intent":"inout", "from_dycore": False})
    qgraupel_t1: FloatField = field(metadata={"name": "physics_graupel_mixing_ratio", "units": "kg/kg", "intent":"inout", "from_dycore": False}) 
    qcld_t1: FloatField  = field(metadata={"name": "physics_cloud_fraction","units": "", "intent":"inout", "from_dycore": False})
    pt_t1: FloatField  = field(metadata={"name": "physics_air_temperature", "units": "degK", "intent":"inout", "from_dycore": False})
    ua_t1: FloatField = field(metadata={"name": "physics_eastward_wind", "units": "m/s", "intent":"inout", "from_dycore": False})
    va_t1: FloatField  = field(metadata={"name": "physics_northward_wind", "units": "m/s", "from_dycore": False })
    delprsi: FloatField = field(metadata={"name": "model_level_pressure_thickness_in_physics", "units": "Pa", "from_dycore": False})
    phii: FloatField = field(metadata={"name": "interface_geopotential_height", "units": "m", "from_dycore": False})
    phil: FloatField = field(metadata={"name": "layer_geopotential_height", "units": "m", "from_dycore": False})
    dz: FloatField = field(metadata={"name": "geopotential_height_thickness", "units": "m", "from_dycore": False})
    wmp: FloatField = field(metadata={"name": "layer_mean_vertical_velocity_microph", "units": "m/s", "from_dycore": False})
    prsi: FloatField = field(metadata={"name": "interface_pressure", "units": "Pa", "from_dycore": False})
    quantity_factory: InitVar[fv3util.QuantityFactory]
  
    def __post_init__(self, quantity_factory):
        # storage for tendency variables not in PhysicsState
        tendency_storage = quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],  "unknown", dtype=float).storage
        self.microphysics = MicrophysicsState(
            pt=self.pt,
            qvapor=self.qvapor,
            qliquid=self.qliquid,
            qrain=self.qrain,
            qice=self.qice,
            qsnow=self.qsnow,
            qgraupel=self.qgraupel,
            qcld=self.qcld,
            ua=self.ua,
            va=self.va,
            delp=self.delp,
            delz=self.delz,
            omga=self.omga,
            delprsi=self.delprsi,
            wmp=self.wmp,
            dz=self.dz,
            tendency_storage=tendency_storage,
        )
        
    @classmethod
    def init_empty(cls, quantity_factory):
        initial_storages = {}
        for field in fields(cls):
            initial_storages[field.name] = quantity_factory.zeros([fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],  field.metadata["units"], dtype=float).storage
        return cls(**initial_storages, quantity_factory=quantity_factory)
    
   
    @classmethod
    def init_from_numpy_arrays(cls, dict_of_numpy_arrays, quantity_factory):
        state = cls.init_empty(quantity_factory)
        field_names = [field.name for field in fields(cls)]
        for variable_name, data in dict_of_numpy_arrays:
            if not variable_name in field_names:
                raise KeyError(variable_name + ' is provided, but not part of the dycore state')
            getattr(state, variable_name).data[:] = data
        for field_name in field_names:
            if not field_name in dict_of_numpy_arrays.keys():
                raise KeyError(field_name + ' is not included in the provided dictionary of numpy arrays')
        return state


    @classmethod
    def init_from_quantities(cls, dict_of_quantities):
        field_names = [field.name for field in fields(cls)]
        for variable_name, data in dict_of_quantities:
            if not variable_name in field_names:
                raise KeyError(variable_name + ' is provided, but not part of the dycore state')
            getattr(state, variable_name).data[:] = data
        for field_name in field_names:
            if not field_name in dict_of_quantities.keys():
                raise KeyError(field_name + ' is not included in the provided dictionary of quantities')
            elif not isinstance(dict_of_quantities[field_name], fv3util.Quantity):
                raise TypeError(field_name + ' is not a Quantity, but instead a ' + type(dict_of_quantities[field_name]))
        return cls(**dict_of_quantities, quantity_factory=None)
