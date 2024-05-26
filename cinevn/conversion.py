"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

import base64
import re
import warnings
from pathlib import Path
from typing import Sequence

import highdicom
import imageio
with warnings.catch_warnings():  # ismrmrd adds a warning filter that causes all subsequent warnings to be printed even if they should be filtered
    import ismrmrd
import numpy as np
import pydicom
import pydicom.dataset
import pydicom.uid
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.exposure import equalize_adapthist

from .data.transforms import apply_affine_to_image


def array2gif(
        image: np.ndarray,
        filename: Path | str,
        clip: bool = True,
        vmin: float | None = None,
        vmax: float | None = None,
        equalize_histogram: bool = False,
        tres: float | None = None,
        file_types: str | Sequence[str] | None = None,
    ) -> None:
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[-1] == 2:
        image = image[..., 0] + 1j * image[..., 1]
    if np.iscomplexobj(image):
        image = np.abs(image)

    # clip, normalize, equalize, convert to uint8
    image = _normalize_image(
        image, clip=clip, vmin=vmin, vmax=vmax, equalize_histogram=equalize_histogram
    )

    _write_files(image, filename, file_types, tres)


def dicom2gif(
        dcm_dir: Path | str | None,
        dcm_dsets: Sequence[pydicom.FileDataset] | None,
        filename: Path | str,
        clip: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        equalize_histogram: bool = False,
        rotation: int = 0,
        flip_horizontal: bool = False,
        flip_vertical: bool = False,
        comment: str = '',
        file_types: str | Sequence[str] | None = None,
    ) -> None:
    if dcm_dir is None and dcm_dsets is None:
        raise ValueError('Either "dcm_dir" or "dcm_dsets" must be provided.')
    if dcm_dir is not None and dcm_dsets is not None:
        raise ValueError('Only one of "dcm_dir" or "dcm_dsets" can be provided.')

    if dcm_dsets is None:
        assert dcm_dir is not None
        dcm_dir = Path(dcm_dir)
        dcm_dsets = [pydicom.dcmread(f) for f in sorted(Path(dcm_dir).glob('*.dcm'))]

    # load DICOM files
    image_arrays: list[tuple[float, np.ndarray]] = []
    tres = None
    for dcm in dcm_dsets:
        dcm_data = dcm.pixel_array
        image_arrays.append((dcm.TriggerTime, dcm_data))
        if tres is None:
            if 'RETRO' in dcm.ImageType or 'INTRPL' in dcm.ImageType:
                # retro-gated or interpolated
                try:
                    nominal_interval = dcm.NominalInterval
                except AttributeError:
                    match = re.search(r'RR \d* \+/-', dcm.ImageComments)
                    if match:
                        nominal_interval = float(match.group()[3:-3])
                    else:
                        nominal_interval = None
                try:
                    number_of_images = dcm.CardiacNumberOfImages
                except AttributeError:
                    number_of_images = len(dcm_dsets)

                if nominal_interval is not None:
                    tres = nominal_interval / number_of_images
                else:
                    tres = dcm.RepetitionTime
            else:
                # triggered
                tres = dcm.RepetitionTime

    # sort arrays and stack to 3D array
    image_arrays = sorted(image_arrays, key=lambda x: x[0])  # sort by trigger time
    cine = np.stack([a[1] for a in image_arrays])

    # clip, normalize, equalize, convert to uint8
    cine = _normalize_image(
        cine, clip=clip, vmin=vmin, vmax=vmax, equalize_histogram=equalize_histogram
    )

    # fix orientation
    cine = apply_affine_to_image(
        cine, rotation=rotation, flip_horizontal=flip_horizontal, flip_vertical=flip_vertical
    )

    # write comments
    if len(comment) > 0:
        cine_commented = []
        for frame in cine:
            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            posx = frame.shape[1] // 2
            posy = frame.shape[0] - 10
            draw.text(
                xy=(posx, posy), text=comment, fill='white', stroke_fill='black', stroke_width=1,
                font=ImageFont.truetype('arial', round(img_pil.height / 15)), anchor='ms'
            )
            cine_commented.append(np.array(img_pil))
        cine = np.stack(cine_commented)

    _write_files(cine, filename, file_types, tres)  # type: ignore


def _normalize_image(image: np.ndarray, clip: bool = False, vmin: float | None = None, vmax: float | None = None,
                    equalize_histogram: bool = False) -> np.ndarray:
    if clip:
        image = np.clip(image, np.percentile(image, 3), np.percentile(image, 97))
    if vmin is None:
        vmin = float(np.min(image))
    if vmax is None:
        vmax = float(np.max(image))
    image = (image - vmin) / (vmax - vmin)
    if equalize_histogram:
        image = equalize_adapthist(image, clip_limit=0.015)
    image *= 255
    image = image.astype(np.uint8)

    return image


def _write_files(image: np.ndarray, filename: Path | str, file_types: str | Sequence[str] | None = None,
                 tres: float | None = 50) -> None:
    filename = Path(filename)
    filename.parent.mkdir(exist_ok=True, parents=True)

    if file_types is None or len(file_types) == 0:
        if filename.suffix != '':
            file_types = [filename.suffix[1:]]
        elif image.ndim == 2:  # static
            file_types = ['png']
        elif image.ndim == 3:  # dynamic
            file_types = ['gif', 'apng']
        else:
            raise ValueError('Could not infer desired file type.')

    for ft in file_types:
        if ft not in ['png', 'gif', 'apng']:
            raise ValueError(f'Unknown file type "{ft}".')

    frame_duration = tres if tres is not None else 50
    if 'png' in file_types:
        if image.ndim == 2:
            imageio.imwrite(filename.with_suffix('.png'), image)
        else:
            for i in range(image.shape[0]):
                imageio.imwrite(filename.with_name(f'{filename.stem}_{i:03d}').with_suffix('.png'), image[i])
    if 'gif' in file_types:
        imageio.v3.imwrite(filename.with_suffix('.gif'), image, duration=frame_duration)
    if 'apng' in file_types:
        imageio.v3.imwrite(filename.with_suffix('.apng'), image, duration=frame_duration)


def array2dicom(
        images: Sequence[np.ndarray],
        mrd_header: ismrmrd.xsd.ismrmrdHeader,
        positions: Sequence[np.ndarray],
        orientations: Sequence[np.ndarray],
        series_number: int = 1,
        instance_number: int = 1,
        series_description_appendix: str = '',
        rot90: bool = False,
        dcm_header_json_b64: str | None = None,
        tags: Sequence[dict] | None = None,
        enhanced: bool = True,
    ) -> pydicom.Dataset | list[pydicom.Dataset]:

    if tags is None:
        tags = [{}] * len(images)
    if 'StudyInstanceUID' not in tags[0]:
        study_instance_uid = pydicom.uid.generate_uid()
        for i in range(len(images)):
            tags[i]['StudyInstanceUID'] = study_instance_uid
    if 'SeriesInstanceUID' not in tags[0]:
        series_instance_uid = pydicom.uid.generate_uid()
        for i in range(len(images)):
            tags[i]['SeriesInstanceUID'] = series_instance_uid

    dcm_dsets = []
    for i in range(len(images)):
        dcm_dsets.append(_create_dicom(
            images[i], mrd_header, positions[i], orientations[i], series_number, instance_number,
            series_description_appendix, rot90, dcm_header_json_b64, **tags[i],
        ))

    if enhanced:
        dcm_dset = highdicom.legacy.LegacyConvertedEnhancedMRImage(
            dcm_dsets, dcm_dsets[0].SeriesInstanceUID, series_number, dcm_dsets[0].SOPInstanceUID, instance_number,
            dcm_dsets[0].file_meta.TransferSyntaxUID
        )
        dcm_dset.NumberOfTemporalPositions = len(images)
        dcm_dset.NumberOfFrames            = len(images)
        return dcm_dset
    else:
        for i in range(len(images)):
            dcm_dsets[i].SeriesNumber = series_number
            dcm_dsets[i].InstanceNumber = instance_number + i
            dcm_dsets[i].NumberOfTemporalPositions = len(images)
        return dcm_dsets


def _create_dicom(
        image: np.ndarray,
        mrd_header: ismrmrd.xsd.ismrmrdHeader,
        position: np.ndarray,
        orientation: np.ndarray,
        series_number: int = 1,
        instance_number: int = 1,
        series_description_appendix: str = '',
        rot90: bool = False,
        dcm_header_json_b64: str | None = None,
        **tags,
    ) -> pydicom.Dataset:

    if dcm_header_json_b64 is None:
        dcm_dset = pydicom.Dataset()
    else:
        dcm_dset = pydicom.Dataset.from_json(base64.b64decode(dcm_header_json_b64))

    dcm_dset.file_meta                            = pydicom.dataset.FileMetaDataset()
    dcm_dset.file_meta.MediaStorageSOPClassUID    = pydicom.uid.MRImageStorage
    dcm_dset.file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
    dcm_dset.file_meta.TransferSyntaxUID          = pydicom.uid.ExplicitVRLittleEndian
    dcm_dset.is_little_endian                     = True
    dcm_dset.is_implicit_VR                       = False
    dcm_dset.ImageType                            = ['ORIGINAL', 'PRIMARY', 'CINE', 'NONE']
    dcm_dset.SOPClassUID                          = dcm_dset.file_meta.MediaStorageSOPClassUID
    dcm_dset.SOPInstanceUID                       = dcm_dset.file_meta.MediaStorageSOPInstanceUID
    dcm_dset.StudyDate                            = '19000101'
    dcm_dset.StudyTime                            = '000000'
    dcm_dset.AccessionNumber                      = '0'
    dcm_dset.Modality                             = 'MR'
    dcm_dset.Manufacturer                         = 'Unknown'
    dcm_dset.PixelPresentation                    = 'MONOCHROME'
    dcm_dset.ComplexImageComponent                = 'MAGNITUDE'
    dcm_dset.PatientName                          = 'xxxx^xxxx'
    dcm_dset.PatientID                            = '0'
    dcm_dset.PatientBirthDate                     = '19000101'
    dcm_dset.PatientSex                           = 'O'
    dcm_dset.SliceThickness                       = 1.
    dcm_dset.ResonantNucleus                      = '1H'
    dcm_dset.StudyInstanceUID                     = pydicom.uid.generate_uid()
    dcm_dset.SeriesInstanceUID                    = pydicom.uid.generate_uid()
    dcm_dset.StudyID                              = '0'
    dcm_dset.SeriesNumber                         = series_number
    dcm_dset.InstanceNumber                       = instance_number
    dcm_dset.ImagePositionPatient                 = list(position)
    dcm_dset.ImageOrientationPatient              = list(orientation[:2].flat)
    dcm_dset.SamplesPerPixel                      = 1
    dcm_dset.PhotometricInterpretation            = 'MONOCHROME2'
    dcm_dset.Rows                                 = image.shape[-2]
    dcm_dset.Columns                              = image.shape[-1]
    dcm_dset.PixelSpacing                         = [1., 1.]
    dcm_dset.BitsAllocated                        = np.array(0, dtype=image.dtype).nbytes * 8
    dcm_dset.BitsStored                           = np.array(0, dtype=image.dtype).nbytes * 8
    dcm_dset.HighBit                              = np.array(0, dtype=image.dtype).nbytes * 8 - 1
    dcm_dset.PixelRepresentation                  = 0  # Unsigned integer
    dcm_dset.WindowCenter                         = image.min() + (image.max() - image.min()) / 2
    dcm_dset.WindowWidth                          = image.max() - image.min()
    dcm_dset.PixelData                            = image.tobytes()

    # fill/overwrite fields with mrd_header
    if mrd_header.acquisitionSystemInformation is not None:
        if mrd_header.acquisitionSystemInformation.systemVendor          is not None: dcm_dset.Manufacturer          = mrd_header.acquisitionSystemInformation.systemVendor
        if mrd_header.acquisitionSystemInformation.institutionName       is not None: dcm_dset.InstitutionName       = mrd_header.acquisitionSystemInformation.institutionName
        if mrd_header.acquisitionSystemInformation.stationName           is not None: dcm_dset.StationName           = mrd_header.acquisitionSystemInformation.stationName
        if mrd_header.acquisitionSystemInformation.systemModel           is not None: dcm_dset.ManufacturerModelName = mrd_header.acquisitionSystemInformation.systemModel
        if mrd_header.acquisitionSystemInformation.systemFieldStrength_T is not None: dcm_dset.MagneticFieldStrength = mrd_header.acquisitionSystemInformation.systemFieldStrength_T
    if mrd_header.experimentalConditions is not None:
        if mrd_header.experimentalConditions.H1resonanceFrequency_Hz     is not None: dcm_dset.ImagingFrequency      = mrd_header.experimentalConditions.H1resonanceFrequency_Hz / 1000000
    if mrd_header.measurementInformation is not None:
        if mrd_header.measurementInformation.protocolName                is not None: dcm_dset.SeriesDescription     = mrd_header.measurementInformation.protocolName + series_description_appendix
        if mrd_header.measurementInformation.protocolName                is not None: dcm_dset.ProtocolName          = mrd_header.measurementInformation.protocolName
        if mrd_header.measurementInformation.patientPosition             is not None: dcm_dset.PatientPosition       = mrd_header.measurementInformation.patientPosition.name
        if mrd_header.measurementInformation.frameOfReferenceUID         is not None: dcm_dset.FrameOfReferenceUID   = mrd_header.measurementInformation.frameOfReferenceUID
    if mrd_header.sequenceParameters is not None:
        if mrd_header.sequenceParameters.TR                              is not None: dcm_dset.RepetitionTime        = mrd_header.sequenceParameters.TR
        if mrd_header.sequenceParameters.TE                              is not None: dcm_dset.EchoTime              = mrd_header.sequenceParameters.TE
    if mrd_header.studyInformation is not None:
        if mrd_header.studyInformation.studyDate                         is not None: dcm_dset.StudyDate             = str(mrd_header.studyInformation.studyDate).replace('-', '')
        if mrd_header.studyInformation.studyTime                         is not None: dcm_dset.StudyTime             = str(mrd_header.studyInformation.studyTime).replace(':', '')
        if mrd_header.studyInformation.accessionNumber                   is not None: dcm_dset.AccessionNumber       = mrd_header.studyInformation.accessionNumber
        if mrd_header.studyInformation.studyDescription                  is not None: dcm_dset.StudyDescription      = mrd_header.studyInformation.studyDescription
        if mrd_header.studyInformation.bodyPartExamined                  is not None: dcm_dset.BodyPartExamined      = mrd_header.studyInformation.bodyPartExamined
        if mrd_header.studyInformation.studyID                           is not None: dcm_dset.StudyID               = mrd_header.studyInformation.studyID
    if mrd_header.subjectInformation is not None:
        if mrd_header.subjectInformation.patientName                     is not None: dcm_dset.PatientName           = mrd_header.subjectInformation.patientName
        if mrd_header.subjectInformation.patientID                       is not None: dcm_dset.PatientID             = mrd_header.subjectInformation.patientID
        if mrd_header.subjectInformation.patientBirthdate                is not None: dcm_dset.PatientBirthDate      = str(mrd_header.subjectInformation.patientBirthdate).replace('-', '')
        if mrd_header.subjectInformation.patientGender                   is not None: dcm_dset.PatientSex            = mrd_header.subjectInformation.patientGender
        if mrd_header.subjectInformation.patientHeight_m                 is not None: dcm_dset.PatientSize           = mrd_header.subjectInformation.patientHeight_m
        if mrd_header.subjectInformation.patientWeight_kg                is not None: dcm_dset.PatientWeight         = mrd_header.subjectInformation.patientWeight_kg
    try:
        fov = mrd_header.encoding[0].reconSpace.fieldOfView_mm  # type: ignore
        div = image.shape[-2:][::-1] if rot90 else image.shape[-2:]
        if True:                                                                      dcm_dset.PixelSpacing          = list(np.array([fov.x, fov.y]) / div)  # type: ignore
        if True:                                                                      dcm_dset.SliceThickness        = fov.z  # type: ignore
    except (AttributeError, IndexError):
        pass

    # fill/overwrite fields with kwargs
    for key, value in tags.items():
        setattr(dcm_dset, key, value)

    pydicom.dataset.validate_file_meta(dcm_dset.file_meta)
    return dcm_dset
