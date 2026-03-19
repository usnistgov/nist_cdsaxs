from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cdsaxs.tools import default_mask, rotate_image, rotate_image_pillow


class DataImage():
    def __init__(self,
                 image: NDArray[np.floating],
                 mask: NDArray[np.bool] = None):
        """
        Generic 2D data class with basic image functionalities. This
        class is not tied to any diffraction information.

        Attributes
        ----------
        image : NDArray
            Two-dimensional array containing the image as pixel
            intensities. The first dimension corresponds to image rows
            from top to bottom and the second dimension corresponds to
            image columns from left to right.
        mask : NDArray, optional
            Two-dimensional boolean array of same dimensions as image
            that are True at pixel values that should be masked out
            for all operations. These pixels will be masked in addition
            to the default masked pixels of nan, inf, -inf. The user
            can overwrite these defaults using the overwrite_mask
            method but we caution against this as not all operations
            are tested without masking nan, inf, and -inf.

        Protected Attributes
        --------------------
        _data_transformations : list of tuples
            Will keep track of intensity data transformations, including
            a normalization, scaling, adding or subtracting by or of a
            specified value. Each item in the list is a tuple of
            (transformation, value) where transformation can be:
                normalize
                scale
                add
                subtract
            and where value can either be a single float or an array of
            floats with the same dimensions as the image.
        _raw_image : NDArray
            Original image provided during initialization of an
            instance of this class. This enables the user to fully
            reset to the original image regardless of any data
            transformations performed.
        _masked_image : NDArray
            Retrieve the current image of the DataImage instance with
            all masked points replaced with np.nan.
        """
        self.image = image
        self._raw_image = np.copy(self.image)
        self.mask = default_mask(self.image)
        if mask is not None:
            self.mask_points(mask)
        self._data_transformations = []

    @property
    def _masked_image(self):
        masked_image = np.copy(self.image)
        masked_image[self.mask] = np.nan
        return masked_image

    def mask_points(self, mask):
        """
        Add points to the data mask. This will not unmask any previously
        masked points in the image.

        Parameters
        ----------
        mask : NDArray
            Two-dimensional boolean array of same dimensions as the
            data image. Pixels that are True will be masked out for
            all data operations. This will NOT unmask any previously
            masked points.
        """
        if mask.shape != self.image.shape:
            raise ValueError(
                "Mask does not match shape of image."
            )
        self.mask += mask

    def _overwrite_mask(self, mask):
        """
        Set a new mask for the data. This will unmask all previously
        masked points and only mask the points provided to this
        function call.

        Be cautious of this operation as other functions assume that
        all nan, inf, -inf points in the original data are always
        masked by default. Unexpected behavior may occur.

        Parameters
        ----------
        mask : NDArray
            Two-dimensional boolean array of same dimensions as the
            data image. Pixels that are True will be masked out for
            all data operations. This will unmask any previously
            masked points.
        """
        self.mask = self.mask*False + mask

    def reset_mask(self, use_raw_image=False):
        """
        Reset to the default mask to only mask out pixels with values of nan,
        inf, or -inf. By default, the current image after any data
        transformations will be used. The raw image can be used by
        switching the keyword argument 'use_raw_image' to True.
        """
        self.mask = default_mask(
            self._raw_image if use_raw_image else self.image
        )

    def rotate_image(self,
                     rotation_angle_deg,
                     rotation_center=(0, 0),
                     resampling_mode="bilinear",
                     fill_mode="constant",
                     fill_constant=np.nan,
                     use_pillow=False,
                     **kwargs):

        """
        Rotate the image counterclockwise by the specified angle about
        the rotation center.
        NOTE: This operation will convert any masked points in your
        array to nan prior to the image rotation so they are not used
        in the resampling algorithms. The mask will then be reset to
        mask out any nan pixels after the rotation.

        Parameters
        ----------
        rotation_angle_deg : float
            Angle in degrees by which to rotate the image
            counterclockwise.
        rotation_center : tuple
            Center of rotation (y, x).
            Default is the upper left pixel.
        resampling_mode : str, optional
            Set the resampling method used during the rotation.
            The box rotation works by rotating the image underneath then
            extracting the box for integration. Resampling modes are
            chosen from the sklearn.transform.warp method. Options are:
                nearest_neighbor
                bilinear (default)
                biquadratic
                bicubic
                biquartic
                biquintic
            Default value is 'bilinear'.
            If use_pillow is set to True, then the options for the
            PILLOW package rotation algorithm are different:
                nearest
                bilinear
                bicubic
        fill_mode : str, optional
            Determine how pixels outside the boundaries of the input image
            are filled after the rotation. Options match those from np.pad.
            Options are:
                constant (default)
                edge
                symmetric
                reflect
                wrap
            Default value is "constant".
        fill_constant : float, optional
            Specifies the constant value used to fill pixels outside the
            image boundaries after rotation. Only applies when resampling_mode
            is set to 'constant'.
        log_scale : bool, optional
            Rotate the log-scale of your image. This could help resolve
            some artifacts caused by certain rotation sampling algorithms
            but you will lose any pixels that are negative (turned to nan).
            Deafult value is False.
        use_pillow : bool, optional
            If set to True, the algorithm will use the PILLOW package
            image rotation function instead of sklearn.transform.rotate.
            The fill_mode argument is not used and the resampling_mode
            options are slightly different, see the above description.
        """

        if not use_pillow:
            self.image = rotate_image(
                self._masked_image,
                degrees=rotation_angle_deg,
                rotation_center=rotation_center,
                resampling_mode=resampling_mode,
                fill_mode=fill_mode,
                fill_constant=fill_constant,
                **kwargs)
        else:
            self.image = rotate_image_pillow(
                self._masked_image,
                degrees=rotation_angle_deg,
                rotation_center=rotation_center,
                resampling_mode=resampling_mode,
                **kwargs)
        self.reset_mask()

    def rotate_image_ccw(self, steps=1):
        """
        Rotate the image counterclockwise by 90 degrees, or by a
        specified number of 90 degree steps.

        The original image can be recalled using reset_image(), but this
        will also undo any other scaling, normalizations, adding, or
        subtracting applied to the image intensity.

        Parameters
        ----------
        steps : int
            Number of 90 degree counterclockwise rotations to be
            performed. If a negative value is provided, the rotations
            will be performed in the clockwise direction.
        """

        k = int(np.round(steps, 0))

        # determine counterclockwise steps to achieve same rotation
        while k < 0:
            k += 4
        if k != 0:
            self.image = np.rot90(self.image, k=k, axes=(0, 1))
            self.mask = np.rot90(self.mask, k=k, axes=(0, 1))

    def flip_horizontally(self):

        self.image = np.flip(self.image, axis=1)
        self.mask = np.flip(self.mask, axis=1)

    def flip_vertically(self):

        self.image = np.flip(self.image, axis=0)
        self.mask = np.flip(self.mask, axis=0)

    def scale_data(self, value):
        """
        Scale the data by the specified value or array of values
        that match the dimensions of the data image.
        """
        if type(value) is float or type(value) is int:
            value = float(value)
        else:
            if value.shape != self.image.shape:
                raise ValueError(
                    "Size of the provided array does not"
                    "match the size of the image data.")

        self.image = self.image*value
        self._data_transformations.append(("scale", value))

    def normalize_data(self, value):
        """
        Scale the data by the recipricol of the specified value.or array
        of values that match the dimensions of the data image.
        """
        if type(value) is float or type(value) is int:
            value = float(value)
            value_r = 1/value
        else:
            value_r = np.reciprocal(value)
            if value_r.shape != self.image.shape:
                raise ValueError(
                    "Size of the provided array does not"
                    "match the size of the image data.")

        self.image = self.image*value_r
        self._data_transformations.append(("normalize", value))

    def subtract_from_data(self, value):
        """
        Subtract a specified single value or an array of values that
        matches the image dimensions from the image data.
        """
        if type(value) is float or type(value) is int:
            value = float(value)
        else:
            if value.shape != self.image.shape:
                raise ValueError(
                    "Size of the provided array does not"
                    "match the size of the image data.")
        self.image = self.image-value
        self._data_transformations.append(("subtract", value))

    def add_to_data(self, value):
        """
        Add a specified single value or an array of values that
        matches the image dimensions to the image data.
        """
        if type(value) is float or type(value) is int:
            value = float(value)
        else:
            if value.shape != self.image.shape:
                raise ValueError(
                    "Size of the provided array does not"
                    "match the size of the image data.")
        self.image = self.image+value
        self._data_transformations.append(("add", value))

    def reset_intensity(self):
        """
        Resets any normailzation, scaling, added or subtracted values
        applied to the image data intensity.
        """
        for transform, value in reversed(self._data_transformations):
            if transform == "add":
                self.subtract_from_data(value)
            elif transform == "subtract":
                self.add_to_data(value)
            elif transform == "normalize":
                self.scale_data(value)
            elif transform == "scale":
                self.normalize_data(value)
        self._data_transformations = []

    def reset_image(self):
        """
        Resets the image to the raw image and also resets the applied
        mask. This will undo any orientation transformations to the
        image as well as any scaling, normalization, additions or
        subtractions applied to the image.
        """
        self._data_transformations = []
        self.image = np.copy(self._raw_image)
        self.reset_mask()

    def sum_box(
            self,
            limits_axis0,
            limits_axis1,
            axis,
    ):
        """
        Select a region of interest (box shape) and sum over the
        selected axis (or axes).

        If any masked values are found along the selected axis, the
        entire column/row will result in a np.nan value.

        Be careful if you have any pixels with a value of nan that are
        not masked by the mask attribute. If the summation algorithm
        encounters all nan values, it will return 0 rather than nan. By
        default, all nan values are masked out unless the user
        overwrites this behavior.

        Parameters
        ----------
        limits_axis0: tuple[int, int]
            Defines the limits (indices) of the box in the first
            dimension. This is a half open range [min, max).
        limits_axis1 : tuple[int, int]
            Defines the limits (indices) of the box in the second
            dimension. This is a half open range [min, max).
        axis : int | tuple
            The axis or axes over which to perform the sum.
            Setting axis to 0 will sum over each row.
            Setting axis to 1 will sum over each column.
            Setting axis to (0, 1) will sum over all axes and return a
            single value.

        Returns
        -------
        sum_intensity : ndarray
            One dimensional array of summed intensity of the defined
            region of interest summed over the selected axis or axes.
        image_box : ndarray
            The two dimensional region of interest selected from the
            data image used in the summation.
        mask_box : ndarray
            The corresponding region of interest selected from the mask
            and used in the summation.
        """
        image_box = self.image[
            limits_axis0[0]:limits_axis0[1], limits_axis1[0]:limits_axis1[1]]
        mask_box = self.mask[
            limits_axis0[0]:limits_axis0[1], limits_axis1[0]:limits_axis1[1]
        ]
        sum_intensity = np.nansum(
            image_box,
            axis=axis,
            where=~mask_box
        ).reshape(-1)
        sum_intensity[mask_box.any(axis=axis)] = np.nan

        return sum_intensity, image_box, mask_box

    def mean_box(
            self,
            limits_axis0,
            limits_axis1,
            axis,
    ):
        """
        Select a region of interest (box shape) and perform an
        arithmetic mean over the selected axis (or axes).

        If any masked values are found along the selected axis, the
        entire column/row will result in a np.nan value.

        Parameters
        ----------
        limits_axis0: tuple[int, int]
            Defines the limits (indices) of the box in the first
            dimension. This is a half open range [min, max).
        limits_axis1 : tuple[int, int]
            Defines the limits (indices) of the box in the second
            dimension. This is a half open range [min, max).
        axis : int | tuple
            The axis or axes over which to perform the mean.
            Setting axis to 0 will average over each row.
            Setting axis to 1 will average over each column.
            Setting axis to (0, 1) will average over all axes and return
            a single value.

        Returns
        -------
        mean_intensity : ndarray
            One dimensional array of mean intensity of the defined
            region of interest over the selected axis or axes.
        image_box : ndarray
            The two dimensional region of interest selected from the
            data image used in the mean operation.
        max_box : ndarray
            The corresponding region of interest selected from the mask
            and used in the mean operation.
        """
        image_box = self.image[
            limits_axis0[0]:limits_axis0[1], limits_axis1[0]:limits_axis1[1]
        ]
        mask_box = self.mask[
            limits_axis0[0]:limits_axis0[1], limits_axis1[0]:limits_axis1[1]
        ]
        mean_intensity = np.nanmean(
            image_box,
            axis=axis,
            where=~mask_box
        )
        mean_intensity[mask_box.any(axis=axis)] = np.nan

        return mean_intensity, image_box, mask_box

    def slice_box(
            self,
            limits_axis0,
            limits_axis1,
            axis,
            mode,
    ):
        """
        Select a region of interest (box shape) and perform an
        arithmetic mean or sum over the selected axis (or axes).

        Parameters
        ----------
        limits_axis0: tuple[int, int]
            Defines the limits (indices) of the box in the first
            dimension. This is a half open range [min, max).
        limits_axis1 : tuple[int, int]
            Defines the limits (indices) of the box in the second
            dimension. This is a half open range [min, max).
        axis : int | tuple
            The axis or axes over which to perform the mean or sum.
            Setting axis to 0 will average/sum over each row.
            Setting axis to 1 will average/sum over each column.
            Setting axis to (0, 1) will average/sum over all axes and return
            a single value.
        mode : str
            Select whether to perform a 'mean' or 'sum'.

        Returns
        -------
        slice_i : ndarray
            One dimensional array of mean or summed intensity of the
            defined region of interest over the selected axis or axes.
        slice_box : ndarray
            The two dimensional region of interest selected from the
            data image used in the mean or sum operation.
        mask_box : ndarray
            The corresponding region of interest selected from the mask
            and used in the mean or sum operation.
        """

        if mode == 'sum':
            slice_i, slice_box, mask_box = self.sum_box(
                limits_axis0=limits_axis0,
                limits_axis1=limits_axis1,
                axis=axis
            )
        elif mode == 'mean':
            slice_i, slice_box, mask_box = self.mean_box(
                limits_axis0=limits_axis0,
                limits_axis1=limits_axis1,
                axis=axis
            )
        else:
            raise ValueError(
                f"Did not recognize slice mode {mode}. Use 'sum' or 'mean'."
            )

        return slice_i, slice_box, mask_box
