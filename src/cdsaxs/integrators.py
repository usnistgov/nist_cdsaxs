"""
Includes integrators for the Dataset class.
"""

from cdsaxs.dataset import Dataset


def integrate_dataset(
        dataset: Dataset,
        limits_qdy_px,
        limits_qdx_px,
        mode,
        axis,
        in_place: bool = True,
        ):
    """
    Most general dataset integration function.

    Parameters
    ----------

    """

    integrated_datas = {}
    for key, data in dataset.datas.items():
        integrated_q_slice = data.integrate_box(
            limits_qdy_px=limits_qdy_px,
            limits_qdx_px=limits_qdx_px,
            mode=mode,
            axis=axis,
        )
        if key != integrated_q_slice.name:
            raise KeyError(
                "Something is wrong, the dataset data keys don't match"
                "the integrated slice names."
            )
        else:
            integrated_datas[key] = integrated_q_slice

    if in_place:
        dataset.integrated_datasets = [integrated_datas]\
            if dataset.integrated_datasets is None\
            else dataset.integrated_datasets.append(integrated_datas)
    else:
        return integrated_datas

import time
from IPython.display import clear_output

def integrate_dataset_box_of_size(
        dataset: Dataset,
        size_qdy_px,
        size_qdx_px,
        mode,
        axis,
        offset_qdy_px=0,
        offset_qdx_px=0,
        in_place=True,
):
    integrated_datas = {}
    i = 0
    for key, data in dataset.datas.items():
        
        integrated_q_slice = data.integrate_box_of_size(
                size_qdy_px=size_qdy_px,
                size_qdx_px=size_qdx_px,
                mode=mode,
                axis=axis,
                offset_qdy_px=offset_qdy_px,
                offset_qdx_px=offset_qdx_px,
                show_plot=False
            )
        
        # if interactive_plot:
        #     print(f"Image {i+1} of {len(dataset.datas.keys())}.")
        #     user_input = input(
        #         "Press Enter to continue, type 'exit' to cancel"
        #         " integration, or type 'complete' to immediately process"
        #         " all images.")

        #     clear_output(wait=True)
        #     if user_input.lower() == 'exit':
        #         print("Integration canceled.")
        #         return None
        #     elif user_input.lower() == 'complete all':
        #         print("Remaining images completed.")
        #         interactive_plot = False

        if key != integrated_q_slice.name:
            raise KeyError(
                "Something is wrong, the dataset data keys don't match"
                "the integrated slice names."
            )
        else:
            integrated_datas[key] = integrated_q_slice    

        i += 1    

    if in_place:
        dataset.integrated_datasets = [integrated_datas]\
            if dataset.integrated_datasets is None\
            else dataset.integrated_datasets.append(integrated_datas)
        return None
    else:
        return integrated_datas


def integrate_dataset_box_of_q_range(
        dataset: Dataset,
        range_qdy,
        range_qdx,
        mode,
        axis,
        in_place=True
):

    integrated_datas = {}
    for key, data in dataset.datas.items():
        integrated_q_slice = data.integrate_box_of_q_range(
            range_qdy=range_qdy,
            range_qdx=range_qdx,
            mode=mode,
            axis=axis,
        )
        if key != integrated_q_slice.name:
            raise KeyError(
                "Something is wrong, the dataset data keys don't match"
                "the integrated slice names."
            )
        else:
            integrated_datas[key] = integrated_q_slice

    if in_place:
        dataset.integrated_datasets = [integrated_datas]\
            if dataset.integrated_datasets is None\
            else dataset.integrated_datasets.append(integrated_datas)
    else:
        return integrated_datas
