import base64
import warnings
from typing import Sequence

import highdicom
with warnings.catch_warnings():  # ismrmrd adds a warning filter that causes all subsequent warnings to be printed even if they should be filtered
    import ismrmrd
import numpy as np
import pydicom
import pydicom.dataset
import pydicom.uid


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
