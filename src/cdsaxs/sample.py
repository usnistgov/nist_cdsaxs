"""
Simple Sample class that holds information about the sample material."
"""

from cdsaxs.metadata import SAMPLE_METADATA_KEYWORDS
# TODO: create special commonly used samples, such as AgBeh or empty


class Sample():
    """
    Simple class to manage metadata specific to the sample material,
    not the measurement, that may be used for corrections during
    data reduction and analysis.

    sample_metadata : dict
        Contains any relevant sample metadata. These are key : value
        pairs where the key must be in the list below and the value is
        formatted depending on requirements of the parameter.
    user_params : dict
        Contains additional user-provided parameters. These may be
        relevant to the user and are shown in the data table of the GUI
        after the required metadata, but are not used for processing
        the data within the GUI and standard workflows. They key can
        be of any format/type desired by the user.

    Metadata Keywords
    -----------------
    sample_size_mm
    substrate_thickness_um
    substrate_attenuation_coeff_um-1
    """

    def __init__(
            self,
            sample_metadata: dict,
            user_params: dict = None
    ):

        self._check_metadata(sample_metadata)
        self.sample_metadata = sample_metadata

        self.user_params = user_params if user_params is not None else {}

    def update_metadata(self, metadata: dict, overwrite: bool = True):
        """
        Add accepted metadata to the class instance. Existing metadata
        parameters can be updated by keeping the overwrite argument
        to True.

        Parameters
        ----------
        metadata : dict
            Key : value pairs of accepted metadata (key) and their
            values. See class docstring for list of accepted keywords.
        overwrite : bool
            If set to True, any metadata provided to this method will
            overwrite the existing value in the instance if it already
            exists in self.metadata.
            Default value is True.
        """
        if self._check_metadata(metadata):
            for key, value in metadata.items():
                if key in self.sample_metadata.keys() and not overwrite:
                    pass
                else:
                    self.sample_metadata[key] = value

    def remove_metadata(self, metadata_keys: list):
        """
        Remove accepted metadata from this instance of the class.

        Parameters
        ----------
        metadata_keys : list
            List of metadata to remove from this class instance.
        """
        if self._check_metadata({key: 0 for key in metadata_keys}):
            self.metadata = {
                key: value for key, value in self.metadata
                if key not in metadata_keys
                }

    def _check_metadata(self, metadata):
        """
        Check if a metadata dictionary contains any unaccepted metadata
        keywords.
        """
        unaccepted_keywords = [
            x for x in metadata.keys() if x not in SAMPLE_METADATA_KEYWORDS
        ]
        if len(unaccepted_keywords) > 0:
            raise ValueError(
                "The following sample metadata keywords are not accepted:\n" +
                f"{unaccepted_keywords}\n" +
                "The following are accepted sample metadata keywords:\n" +
                f"{SAMPLE_METADATA_KEYWORDS}"
            )

        return True

    def update_user_params(self, params: dict, overwrite: bool = True):
        """
        Add key: value pairs to the user params of this class instance.
        Existing parameters can be updated by keeping the overwrite
        argument as True.

        Parameters
        ----------
        params : dict
            Key : value pairs of user-specified parameters for this
            data instance.
        overwrite : bool
            If set to True, any parameters provided to this method will
            overwrite the existing value in this instance if it already
            exists in self.user_params.
            Default value is True.
        """
        for key, value in params.items():
            if key in self.user_params.keys() and not overwrite:
                pass
            else:
                self.user_params[key] = value

    def remove_user_params(self, param_keys: list):
        """
        Remove the identified parameters from user params of this
        class instance.

        Parameters
        ----------
        param_keys : list
            List of parameters to remove from user_params of this class
            instance.
        """
        self.user_params = {
            key: value for key, value in self.user_params
            if key not in param_keys
            }
