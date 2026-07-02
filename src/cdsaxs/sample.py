"""
Simple Sample class that holds information about the sample material."
"""
import numpy as np

# TODO: create special commonly used samples, such as AgBeh or empty


class Sample():

    def __init__(
            self,
            name=None,
            sample_metadata: dict = None,
            user_params: dict = None,
            known_material: str = None
    ):
        """
        Simple class to manage metadata specific to the sample material,
        not the measurement, that may be used for corrections during
        data reduction and analysis.

        name : str, optional
            Identifier for the sample.
        sample_metadata : dict, optional
            Contains any relevant sample metadata. These are key : value
            pairs where the key must be in the list below and the value is
            formatted depending on requirements of the parameter.
        user_params : dict, optional
            Contains additional user-provided parameters. These may be
            relevant to the user and are shown in the data table of the GUI
            after the required metadata, but are not used for processing
            the data within the GUI and standard workflows. They key can
            be of any format/type desired by the user.
        known_material: str, optional
            A known material can be selected from the following list:
                nist_q_calibration (100 nm pitch)
                agbeh or silver_behenate
            Doing so will populate any known metadata, such as the
            pitch or q peak positions.

        Metadata Keywords
        -----------------
        sample_size_mm
        substrate_thickness_um
        substrate_attenuation_coeff_um-1
        pitch_nm
        q_peak_positions
        """
        pass
#         self.metadata = {}
#         if sample_metadata is not None:
#             self.update_metadata(metadata=sample_metadata, overwrite=True)

#         self.user_params = {}
#         if user_params is not None:
#             self.update_user_params(params=user_params, overwrite=True)

#         if known_material is not None:
#             if known_material.lower() in [
#                 'nist_q_calibration', 'nistqcalibration', 'nist_qcalibration'
#             ]:
#                 self.update_metadata(
#                     metadata=_nist_qcalibration_metadata()
#                 )
#                 self.name = 'nist_q_calibration'
#             elif known_material.lower() in [
#                 'silver_behenate', 'agbeh', 'ag_beh', 'silverbehenate'
#             ]:
#                 self.update_metadata(
#                     metadata=_agbeh_metadata()
#                 )
#                 self.name = 'agbeh'
#             else:
#                 warnings.warn(
#                     f"The known material {known_material} was not recognized."
#                 )
#                 self.name = known_material
#         else:
#             self.name = name

#     def update_metadata(self, metadata: dict, overwrite: bool = True):
#         """
#         Add accepted metadata to the class instance. Existing metadata
#         parameters can be updated by keeping the overwrite argument
#         to True.

#         Parameters
#         ----------
#         metadata : dict
#             Key : value pairs of accepted metadata (key) and their
#             values. See class docstring for list of accepted keywords.
#         overwrite : bool
#             If set to True, any metadata provided to this method will
#             overwrite the existing value in the instance if it already
#             exists in self.metadata.
#             Default value is True.
#         """
#         if check_metadata(metadata, sample_mode=True):
#             for key, value in metadata.items():
#                 if key in self.sample_metadata.keys() and not overwrite:
#                     pass
#                 else:
#                     self.sample_metadata[key] = value

#     def remove_metadata(self, metadata_keys: list):
#         """
#         Remove accepted metadata from this instance of the class.

#         Parameters
#         ----------
#         metadata_keys : list
#             List of metadata to remove from this class instance.
#         """
#         if check_metadata({key: 0 for key in metadata_keys}):
#             self.metadata = {
#                 key: value for key, value in self.metadata
#                 if key not in metadata_keys
#                 }

#     def update_user_params(self, params: dict, overwrite: bool = True):
#         """
#         Add key: value pairs to the user params of this class instance.
#         Existing parameters can be updated by keeping the overwrite
#         argument as True.

#         Parameters
#         ----------
#         params : dict
#             Key : value pairs of user-specified parameters for this
#             data instance.
#         overwrite : bool
#             If set to True, any parameters provided to this method will
#             overwrite the existing value in this instance if it already
#             exists in self.user_params.
#             Default value is True.
#         """
#         for key, value in params.items():
#             if key in self.user_params.keys() and not overwrite:
#                 pass
#             else:
#                 self.user_params[key] = value

#     def remove_user_params(self, param_keys: list):
#         """
#         Remove the identified parameters from user params of this
#         class instance.

#         Parameters
#         ----------
#         param_keys : list
#             List of parameters to remove from user_params of this class
#             instance.
#         """
#         self.user_params = {
#             key: value for key, value in self.user_params
#             if key not in param_keys
#             }


# def _agbeh_metadata():

#     q_peaks = [
#         0.1076,
#         0.2152,
#         0.3228,
#         0.4304,
#         0.5380,
#         0.6456,
#         0.7532,
#         0.8608,
#         0.9684,
#         1.076,
#         1.184,
#     ]

#     metadata = {
#         "q_peak_positions_Ang-1": q_peaks
#     }

#     return metadata


# def _nist_qcalibration_metadata():

#     pitch_nm = 100
#     q_peaks = 2*np.pi*np.arange(1, 11, 1)/(pitch_nm*10)

#     metadata = {
#         'pitch_nm': 100,
#         'q_peak_positions_Ang-1': q_peaks
#     }

#     return metadata
