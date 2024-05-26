#  Copyright (c) 2023. Marc Vornehm <marc.vornehm@fau.de>.

import math
import re
import warnings

with warnings.catch_warnings():  # ismrmrd adds a warning filter that causes all subsequent warnings to be printed even if they should be filtered
    import ismrmrd
import numpy as np
from mapvbvd.read_twix_hdr import twix_hdr
from mapvbvd.twix_map_obj import twix_map_obj

from .enums import Trajectory


def twix_to_ismrmrd(twix_header: twix_hdr, twix_image: twix_map_obj) -> ismrmrd.xsd.ismrmrdHeader:
    """
    Based on https://github.com/ismrmrd/siemens_to_ismrmrd/blob/master/parameter_maps/IsmrmrdParameterMap_Siemens.xsl
    """
    ismrmrd_header = ismrmrd.xsd.ismrmrdHeader(
        subjectInformation=_parse_subject_information(twix_header),
        studyInformation=_parse_study_information(twix_header),
        measurementInformation=_parse_measurement_information(twix_header),
        acquisitionSystemInformation=_parse_acquisition_system_information(twix_header),
        experimentalConditions=_parse_experimental_conditions(twix_header),
        encoding=[_parse_encoding(twix_header, twix_image)],
        sequenceParameters=_parse_sequence_parameters(twix_header),
        userParameters=_parse_user_parameters(twix_header),
    )

    return ismrmrd_header


def _parse_subject_information(hdr: twix_hdr) -> ismrmrd.xsd.subjectInformationType:
    # patientBirthdate
    if hdr['Meas']['PatientBirthDay'].replace('-', '').isnumeric():
        patient_birthdate_str = hdr['Meas']['PatientBirthDay']
    else:  # anonymized
        patient_birthdate_str = '1900-01-01'
    patient_birthdate = ismrmrd.xsd.ismrmrdschema.ismrmrd.XmlDate.from_string(patient_birthdate_str)

    # patientGender
    if hdr['Dicom']['lPatientSex'] == 1:
        patient_gender = 'F'
    elif hdr['Dicom']['lPatientSex'] == 2:
        patient_gender = 'M'
    else:
        patient_gender = 'O'

    subject_information = ismrmrd.xsd.subjectInformationType(
        patientName=hdr['Dicom']['tPatientName'],
        patientWeight_kg=hdr['Meas']['flUsedPatientWeight'],
        patientHeight_m=hdr['Meas']['flPatientHeight'] / 1000,
        patientID=hdr['Meas']['PatientID'],
        patientBirthdate=patient_birthdate,
        patientGender=patient_gender,
    )

    return subject_information


def _parse_study_information(hdr: twix_hdr) -> ismrmrd.xsd.studyInformationType:
    study_information = ismrmrd.xsd.studyInformationType(
        bodyPartExamined=hdr['Dicom']['tBodyPartExamined'],
        studyDescription=hdr['Dicom']['tStudyDescription'],
        studyInstanceUID=hdr['Meas']['StudyLOID'].split('.')[-1],
    )

    return study_information


def _parse_measurement_information(hdr: twix_hdr) -> ismrmrd.xsd.measurementInformationType:
    # measurementID
    patient_id = hdr['Meas']['PatientLOID']
    if patient_id == '':
        patient_id = '10000000'
    study_id = hdr['Meas']['StudyLOID'].split('.')[-1]
    measurement_id = f'{int(hdr["Dicom"]["DeviceSerialNumber"])}.{patient_id}.{study_id}.{int(hdr["Meas"]["MeasUID"])}'

    # relativeTablePosition
    relative_table_position_x = hdr['Dicom']['lGlobalTablePosSag']
    if relative_table_position_x == '':
        relative_table_position_x = 0
    relative_table_position_y = hdr['Dicom']['lGlobalTablePosCor']
    if relative_table_position_y == '':
        relative_table_position_y = 0
    relative_table_position_z = hdr['Dicom']['lGlobalTablePosTra']
    if relative_table_position_z == '':
        relative_table_position_z = 0
    relative_table_position = ismrmrd.xsd.threeDimensionalFloat(
        relative_table_position_x,
        relative_table_position_y,
        relative_table_position_z,
    )

    # measurementDependency
    measurement_dependency = [ismrmrd.xsd.measurementDependencyType(measurementID=x)
                              for x in hdr['Meas']['ReconMeasDependencies'].split(' ') if x.isnumeric()]

    measurement_information = ismrmrd.xsd.measurementInformationType(
        measurementID=measurement_id,
        patientPosition=ismrmrd.xsd.patientPositionType(hdr['Meas']['tPatientPosition']),
        relativeTablePosition=relative_table_position,
        protocolName=hdr['Meas']['tProtocolName'],
        measurementDependency=measurement_dependency,
        frameOfReferenceUID=hdr['Meas']['tFrameOfReference'],
    )

    return measurement_information


def _parse_acquisition_system_information(hdr: twix_hdr) -> ismrmrd.xsd.acquisitionSystemInformationType:
    # coilLabel
    coil_label = []
    try:
        position = 0
        while True:
            coil_number = int(hdr['MeasYaps'][('sCoilSelectMeas', 'aRxCoilSelectData', '0', 'asList', str(position),
                                               'lADCChannelConnected')])
            t_coil_id = hdr['MeasYaps'][('sCoilSelectMeas', 'aRxCoilSelectData', '0', 'asList', str(position),
                                         'sCoilElementID', 'tCoilID')].strip('"')
            t_element = hdr['MeasYaps'][('sCoilSelectMeas', 'aRxCoilSelectData', '0', 'asList', str(position),
                                         'sCoilElementID', 'tElement')].strip('"')
            l_coil_copy = int(hdr['MeasYaps'][('sCoilSelectMeas', 'aRxCoilSelectData', '0', 'asList', str(position),
                                               'sCoilElementID', 'lCoilCopy')])
            coil_name = f'{t_coil_id}:{l_coil_copy}:{t_element}'
            coil_label.append(ismrmrd.xsd.coilLabelType(coilNumber=coil_number, coilName=coil_name))
            position += 1
    except KeyError:  # we don't know how many coil labels there are, so at some point we will get a KeyError
        pass

    acquisition_system_information = ismrmrd.xsd.acquisitionSystemInformationType(
        systemVendor=hdr['Dicom']['Manufacturer'],
        systemModel=hdr['Dicom']['ManufacturersModelName'],
        systemFieldStrength_T=hdr['Meas']['flMagneticFieldStrength'],
        relativeReceiverNoiseBandwidth=0.793,
        receiverChannels=int(hdr['Meas']['iMaxNoOfRxChannels']),
        coilLabel=coil_label,
        institutionName=hdr['Dicom']['InstitutionName'],
        deviceID=str(int(hdr['Dicom']['DeviceSerialNumber'])),
    )

    return acquisition_system_information


def _parse_experimental_conditions(hdr: twix_hdr) -> ismrmrd.xsd.experimentalConditionsType:
    experimental_conditions = ismrmrd.xsd.experimentalConditionsType(
        H1resonanceFrequency_Hz=int(hdr['Dicom']['lFrequency']),
    )

    return experimental_conditions


def _parse_encoding(hdr: twix_hdr, img: twix_map_obj) -> ismrmrd.xsd.encodingType:
    # trajectory
    if hdr['Meas']['ucTrajectory'] == Trajectory.TRAJECTORY_CARTESIAN.value:
        trajectory = ismrmrd.xsd.trajectoryType('cartesian')
    elif hdr['Meas']['ucTrajectory'] == Trajectory.TRAJECTORY_RADIAL.value:
        trajectory = ismrmrd.xsd.trajectoryType('radial')
    elif hdr['Meas']['ucTrajectory'] == Trajectory.TRAJECTORY_SPIRAL.value:
        trajectory = ismrmrd.xsd.trajectoryType('spiral')
    # elif hdr['Meas']['ucTrajectory'] == Trajectory.TRAJECTORY_BLADE.value:
    #     trajectory = ismrmrd.xsd.trajectoryType('propellor')
    else:
        trajectory = ismrmrd.xsd.trajectoryType('other')

    # trajectoryDescription
    if hdr['Meas']['ucTrajectory'] == Trajectory.TRAJECTORY_SPIRAL.value:
        trajectory_description = ismrmrd.xsd.trajectoryDescriptionType(
            identifier='HargreavesVDS2000',
            userParameterLong=[
                ismrmrd.xsd.userParameterLongType('interleaves', int(hdr['Meas']['lRadialViews'])),
                ismrmrd.xsd.userParameterLongType('fov_coefficients', 1),
                ismrmrd.xsd.userParameterLongType('SamplingTime_ns',
                                                  int(hdr['MeasYaps'][('sWipMemBlock', 'alFree[57]')])),
            ],
            userParameterDouble=[
                ismrmrd.xsd.userParameterDoubleType('MaxGradient_G_per_cm',
                                                    int(hdr['MeasYaps'][('sWipMemBlock', 'adFree[7]')])),
                ismrmrd.xsd.userParameterDoubleType('MaxSlewRate_G_per_cm_per_s',
                                                    int(hdr['MeasYaps'][('sWipMemBlock', 'adFree[8]')])),
                ismrmrd.xsd.userParameterDoubleType('FOVCoeff_1_cm',
                                                    int(hdr['MeasYaps'][('sWipMemBlock', 'adFree[10]')])),
                ismrmrd.xsd.userParameterDoubleType('krmax_per_cm',
                                                    int(hdr['MeasYaps'][('sWipMemBlock', 'adFree[9]')])),
            ],
            comment='Using spiral design by Brian Hargreaves (http://mrsrl.stanford.edu/~brian/vdspiral/)',
        )
    elif int(hdr['Meas']['alRegridRampupTime'].split(' ')[0]) > 0 and \
            int(hdr['Meas']['alRegridRampdownTime'].split(' ')[0]) > 0:
        trajectory_description = ismrmrd.xsd.trajectoryDescriptionType(
            identifier='ConventionalEPI',
            userParameterLong=[
                ismrmrd.xsd.userParameterLongType('etl', int(hdr['Meas']['lEPIFactor'])),
                ismrmrd.xsd.userParameterLongType('numberOfNavigators', 3),
                ismrmrd.xsd.userParameterLongType('rampUpTime', int(hdr['Meas']['alRegridRampupTime'].split(' ')[0])),
                ismrmrd.xsd.userParameterLongType('rampDownTime',
                                                  int(hdr['Meas']['alRegridRampdownTime'].split(' ')[0])),
                ismrmrd.xsd.userParameterLongType('flatTopTime', int(hdr['Meas']['alRegridFlattopTime'].split(' ')[0])),
                ismrmrd.xsd.userParameterLongType('echoSpacing', int(hdr['Meas']['lEchoSpacing'])),
                ismrmrd.xsd.userParameterLongType('acqDelayTime',
                                                  int(hdr['Meas']['alRegridDelaySamplesTime'].split(' ')[0])),
                ismrmrd.xsd.userParameterLongType('numSamples', int(hdr['Meas']['alRegridDestSamples'].split(' ')[0])),
            ],
            userParameterDouble=[
                ismrmrd.xsd.userParameterDoubleType('dwellTime', int(hdr['Meas']['alDwellTime'].split(' ')[0]) // 1000),
            ],
            comment='Conventional 2D EPI sequence',
        )
    else:
        trajectory_description = None

    # encodedSpace/matrixSize
    if hdr['Meas']['ucTrajectory'] == Trajectory.TRAJECTORY_CARTESIAN.value:
        enc_x = int(hdr['Meas']['iNoOfFourierColumns'])
    else:
        enc_x = int(hdr['Config']['ImageColumns'])
    if 'uc2DInterpolation' in hdr['Meas'] and hdr['Meas']['uc2DInterpolation'] in [1, 'true']:
        enc_y = int(hdr['Meas']['iPEFTLength'] // 2)
    else:
        enc_y = int(hdr['Meas']['iPEFTLength'])
    if 'iNoOfFourierPartitions' not in hdr['Meas']:
        enc_z = 1
    else:
        enc_z = int(hdr['Meas']['i3DFTLength'])
    enc_matrix_size = ismrmrd.xsd.matrixSizeType(enc_x, enc_y, enc_z)

    # encodedSpace/fieldOfView_mm
    if hdr['Meas']['ucTrajectory'] == Trajectory.TRAJECTORY_CARTESIAN.value:
        enc_fov_x = int(hdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'dReadoutFOV')] *
                        hdr['Meas']['flReadoutOSFactor'])
    else:
        enc_fov_x = int(hdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'dReadoutFOV')])
    if 'phaseOversampling' in hdr['Config'] and hdr['Config']['phaseOversampling'] != '' \
            and not math.isnan(hdr['Config']['phaseOversampling']):
        phase_oversampling = hdr['Config']['phaseOversampling']
    else:
        phase_oversampling = 0
    enc_fov_y = hdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'dPhaseFOV')] * (1 + phase_oversampling)
    if 'dSliceOversamplingForDialog' in hdr['Meas'] and hdr['Meas']['dSliceOversamplingForDialog'] != '' \
            and not math.isnan(hdr['Meas']['dSliceOversamplingForDialog']):
        slice_oversampling = hdr['Meas']['dSliceOversamplingForDialog']
    else:
        slice_oversampling = 0
    enc_fov_z = hdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'dThickness')] * (1 + slice_oversampling)
    enc_field_of_view_mm = ismrmrd.xsd.fieldOfViewMm(enc_fov_x, enc_fov_y, enc_fov_z)

    encoded_space = ismrmrd.xsd.encodingSpaceType(matrixSize=enc_matrix_size, fieldOfView_mm=enc_field_of_view_mm)

    # reconSpace/matrixSize
    rec_x = int(hdr['Config']['ImageColumns'])
    rec_y = int(hdr['Config']['ImageLines'])
    if hdr['Meas']['i3DFTLength'] == 1:
        rec_z = 1
    else:
        rec_z = int(hdr['Meas']['lImagesPerSlab'])
    rec_matrix_size = ismrmrd.xsd.matrixSizeType(rec_x, rec_y, rec_z)

    # reconSpace/fieldOfView_mm
    rec_fov_x = hdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'dReadoutFOV')]
    rec_fov_y = hdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'dPhaseFOV')]
    rec_fov_z = hdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'dThickness')]
    rec_field_of_view_mm = ismrmrd.xsd.fieldOfViewMm(rec_fov_x, rec_fov_y, rec_fov_z)

    recon_space = ismrmrd.xsd.encodingSpaceType(matrixSize=rec_matrix_size, fieldOfView_mm=rec_field_of_view_mm)

    # encodingLimits/kspace_encoding_step_1
    assert isinstance(img.centerLin, np.ndarray)
    assert len(np.unique(img.centerLin)) == 1
    kspace_encoding_step_1 = ismrmrd.xsd.limitType(
        minimum=0,
        maximum=int(hdr['Meas']['iNoOfFourierLines'] - 1),
        # center=int(hdr['MeasYaps'][('sKSpace', 'lPhaseEncodingLines')] // 2),
        center=int(img.centerLin[0]),
    )

    # encodingLimits/kspace_encoding_step_2
    if 'iNoOfFourierPartitions' not in hdr['Meas'] or hdr['Meas']['i3DFTLength'] == 1:
        kspace_encoding_step_2_maximum = 0
        kspace_encoding_step_2_center = 0
    else:
        kspace_encoding_step_2_maximum = int(hdr['Meas']['iNoOfFourierPartitions'] - 1)
        if hdr['Meas']['ucTrajectory'] == Trajectory.TRAJECTORY_CARTESIAN.value:
            if ('sPat', 'lAccelFact3D') in hdr['MeasYaps']:
                if not hdr['MeasYaps'][('sPat', 'lAccelFact3D')] > 1:
                    kspace_encoding_step_2_center = int(
                        hdr['MeasYaps'][('sKSpace', 'lPartitions')] // 2
                        - (hdr['Meas']['lPartitions'] - hdr['Meas']['iNoOfFourierPartitions'])
                    )
                elif hdr['MeasYaps'][('sKSpace', 'lPartitions')] - hdr['Meas']['iNoOfFourierPartitions'] > \
                        hdr['MeasYaps'][('sPat', 'lAccelFact3D')]:
                    kspace_encoding_step_2_center = int(
                        hdr['MeasYaps'][('sKSpace', 'lPartitions')] // 2
                        - (hdr['MeasYaps'][('sKSpace', 'lPartitions')] - hdr['Meas']['iNoOfFourierPartitions'])
                    )
                else:
                    kspace_encoding_step_2_center = int(hdr['MeasYaps'][('sKSpace', 'lPartitions')] // 2)
            else:
                kspace_encoding_step_2_center = int(
                    hdr['MeasYaps'][('sKSpace', 'lPartitions')] // 2
                    - (hdr['MeasYaps'][('sKSpace', 'lPartitions')] - hdr['Meas']['iNoOfFourierPartitions'])
                )
        else:
            kspace_encoding_step_2_center = 0
    kspace_encoding_step_2 = ismrmrd.xsd.limitType(
        minimum=0,
        maximum=kspace_encoding_step_2_maximum,
        center=kspace_encoding_step_2_center,
    )

    # encodingLimits/slice
    encoding_limits_slice = ismrmrd.xsd.limitType(
        minimum=0,
        maximum=int(hdr['MeasYaps'][('sSliceArray', 'lSize')] - 1),
        center=0,
    )

    # encodingLimits/set
    encoding_limits_set = ismrmrd.xsd.limitType(
        minimum=0,
        maximum=int(hdr['Meas'].get('iNSet', 1) - 1),
        center=0,
    )

    # encodingLimits/phase
    phase = ismrmrd.xsd.limitType(
        minimum=0,
        maximum=int(hdr['MeasYaps'].get(('sPhysioImaging', 'lPhases'), 1) - 1),
        center=0,
    )

    # encodingLimits/repetition
    repetition = ismrmrd.xsd.limitType(
        minimum=0,
        maximum=int(hdr['MeasYaps'].get(('lRepetitions', ), 0)),
        center=0,
    )

    # encodingLimits/segment
    # segment_maximum = 0
    # if ('sFastImaging', 'ucSegmentationMode') in hdr['MeasYaps']:
    #     if hdr['MeasYaps'][('sFastImaging', 'ucSegmentationMode')] == 2 \
    #             and ('sFastImaging', 'lShots') in hdr['MeasYaps']:
    #         segment_maximum = int(hdr['MeasYaps'][('sFastImaging', 'lShots')] - 1)
    #     elif hdr['MeasYaps'][('sFastImaging', 'ucSegmentationMode')] == 1 \
    #             and hdr['MeasYaps'][('sFastImaging', 'lSegments')] > 1:
    #         segment_maximum = math.ceil(hdr['Meas']['iNoOfFourierPartitions'] * hdr['Meas']['iNoOfFourierLines']
    #                                     / hdr['MeasYaps'][('sFastImaging', 'lSegments')])
    segment_maximum = int(hdr['Meas']['lSegments'] - 1)
    segment = ismrmrd.xsd.limitType(
        minimum=0,
        maximum=segment_maximum,
        center=0,
    )

    # encodingLimits/contrast
    # if '('lContrasts', )' in hdr['MeasYaps']:
    #     contrast_maximum = int(hdr['MeasYaps']['('lContrasts', )'] - 1)
    # else:
    #     contrast_maximum = 0
    contrast_maximum = int(hdr['Meas']['lContrasts'] - 1)
    contrast = ismrmrd.xsd.limitType(
        minimum=0,
        maximum=contrast_maximum,
        center=0,
    )

    # encodingLimits/average
    # if 'lAverages' in hdr['MeasYaps']:
    #     average_maximum = int(hdr['MeasYaps'][('lAverages', )] - 1)
    # else:
    #     average_maximum = 0
    average_maximum = int(hdr['Meas']['lAverages'] - 1)
    average = ismrmrd.xsd.limitType(
        minimum=0,
        maximum=average_maximum,
        center=0,
    )

    encoding_limits = ismrmrd.xsd.encodingLimitsType(
        kspace_encoding_step_1=kspace_encoding_step_1,
        kspace_encoding_step_2=kspace_encoding_step_2,
        average=average,
        slice=encoding_limits_slice,
        contrast=contrast,
        phase=phase,
        repetition=repetition,
        set=encoding_limits_set,
        segment=segment,
    )

    # parallelImaging
    acceleration_factor = ismrmrd.xsd.accelerationFactorType(
        kspace_encoding_step_1=int(hdr['MeasYaps'].get(('sPat', 'lAccelFactPE'), 1)),
        kspace_encoding_step_2=int(hdr['MeasYaps'].get(('sPat', 'lAccelFact3D'), 1)),
    )
    if hdr['MeasYaps'][('sPat', 'ucRefScanMode')] == 1:
        calibration_mode = ismrmrd.xsd.calibrationModeType('other')
        interleaving_dimension = ismrmrd.xsd.interleavingDimensionType('other')
    elif hdr['MeasYaps'][('sPat', 'ucRefScanMode')] == 2:
        calibration_mode = ismrmrd.xsd.calibrationModeType('embedded')
        interleaving_dimension = None
    elif hdr['MeasYaps'][('sPat', 'ucRefScanMode')] == 4:
        calibration_mode = ismrmrd.xsd.calibrationModeType('separate')
        interleaving_dimension = None
    elif hdr['MeasYaps'][('sPat', 'ucRefScanMode')] == 8:
        calibration_mode = ismrmrd.xsd.calibrationModeType('separate')
        interleaving_dimension = None
    elif hdr['MeasYaps'][('sPat', 'ucRefScanMode')] == 16:
        calibration_mode = ismrmrd.xsd.calibrationModeType('interleaved')
        interleaving_dimension = ismrmrd.xsd.interleavingDimensionType('average')
    elif hdr['MeasYaps'][('sPat', 'ucRefScanMode')] == 32:
        calibration_mode = ismrmrd.xsd.calibrationModeType('interleaved')
        interleaving_dimension = ismrmrd.xsd.interleavingDimensionType('repetition')
    elif hdr['MeasYaps'][('sPat', 'ucRefScanMode')] == 64:
        calibration_mode = ismrmrd.xsd.calibrationModeType('interleaved')
        interleaving_dimension = ismrmrd.xsd.interleavingDimensionType('phase')
    else:
        calibration_mode = ismrmrd.xsd.calibrationModeType('other')
        interleaving_dimension = None
    parallel_imaging = ismrmrd.xsd.parallelImagingType(
        accelerationFactor=acceleration_factor,
        calibrationMode=calibration_mode,
        interleavingDimension=interleaving_dimension,
    )

    encoding = ismrmrd.xsd.encodingType(
        encodedSpace=encoded_space,
        reconSpace=recon_space,
        encodingLimits=encoding_limits,
        trajectory=trajectory,
        trajectoryDescription=trajectory_description,
        parallelImaging=parallel_imaging,
    )

    return encoding


def _parse_sequence_parameters(hdr: twix_hdr) -> ismrmrd.xsd.sequenceParametersType:
    # TR
    tr_str = re.split(' +', hdr['Meas']['alTR'])
    tr = [int(tr_str[0]) / 1000]
    tr.extend([int(x) / 1000 for x in tr_str[1:] if int(x) > 0])

    # TE
    te_str = re.split(' +', hdr['Meas']['alTE'])
    te = [int(te_str[0]) / 1000]
    te.extend([int(x) / 1000 for x in te_str[1:min(int(hdr['Meas']['lContrasts']), len(te_str))] if int(x) > 0])

    # TI
    ti_str = re.split(' +', hdr['Meas']['alTI'])
    ti = [int(x) / 1000 for x in ti_str if int(x) > 0]

    # flipAngle_deg
    flip_angle_deg_str = re.split(' +', hdr['Meas']['adFlipAngleDegree'])
    flip_angle_deg = [float(x) for x in flip_angle_deg_str if int(x) > 0]

    # sequence_type
    if 'ucSequenceType' not in hdr['Meas']:
        sequence_type = None
    elif hdr['Meas']['ucSequenceType'] == 1:
        sequence_type = 'Flash'
    elif hdr['Meas']['ucSequenceType'] == 2:
        sequence_type = 'SSFP'
    elif hdr['Meas']['ucSequenceType'] == 4:
        sequence_type = 'EPI'
    elif hdr['Meas']['ucSequenceType'] == 8:
        sequence_type = 'TurboSpinEcho'
    elif hdr['Meas']['ucSequenceType'] == 16:
        sequence_type = 'ChemicalShiftImaging'
    elif hdr['Meas']['ucSequenceType'] == 32:
        sequence_type = 'FID'
    else:
        sequence_type = 'Unknown'

    # echo_spacing
    if 'lEchoSpacing' in hdr['Meas'] and hdr['Meas']['lEchoSpacing'] != '':
        echo_spacing = [int(hdr['Meas']['lEchoSpacing']) / 1000]
    else:
        echo_spacing = []

    sequence_parameters = ismrmrd.xsd.sequenceParametersType(
        TR=tr,
        TE=te,
        TI=ti,
        flipAngle_deg=flip_angle_deg,
        sequence_type=sequence_type,
        echo_spacing=echo_spacing,
    )

    return sequence_parameters


def _parse_user_parameters(hdr: twix_hdr) -> ismrmrd.xsd.userParametersType:
    # userParameterLong
    user_parameter_long = []
    if ('sWipMemBlock', 'alFree') in hdr['MeasYaps']:
        for i, al_free_str in enumerate(re.split(' +', hdr['MeasYaps'][('sWipMemBlock', 'alFree')])):
            user_parameter_long.append(ismrmrd.xsd.userParameterLongType(
                name=f'sWipMemBlock.alFree[{i}]',
                value=int(al_free_str),
            ))
    if ('sAngio', 'sFlowArray', 'lSize') in hdr['MeasYaps']:
        user_parameter_long.append(ismrmrd.xsd.userParameterLongType(
            name='VENC_0',
            value=int(hdr['MeasYaps'][('sAngio', 'sFlowArray', 'asElm', 's0', 'nVelocity')]),
        ))
        user_parameter_long.append(ismrmrd.xsd.userParameterLongType(
            name='Flow_Dir',
            value=int(hdr['MeasYaps'][('sAngio', 'sFlowArray', 'asElm', 's0', 'nDir')]),
        ))
    if hdr['MeasYaps'][('sPhysioImaging', 'lSignal1')] not in [1, 16] \
            and hdr['MeasYaps'][('sPhysioImaging', 'lMethod1')] == 8 \
            and hdr['MeasYaps'][('sFastImaging', 'lShots')] >= 1 \
            and hdr['MeasYaps'][('sPhysioImaging', 'lPhases')] > 1 \
            and hdr['MeasYaps'][('sPhysioImaging', 'lRetroGatedImages')] > 0:
        user_parameter_long.append(ismrmrd.xsd.userParameterLongType(
            name='RetroGatedImages',
            value=int(hdr['MeasYaps'][('sPhysioImaging', 'lRetroGatedImages')]),
        ))
        user_parameter_long.append(ismrmrd.xsd.userParameterLongType(
            name='RetroGatedSegmentSize',
            value=int(hdr['MeasYaps'].get(('sFastImaging', 'lSegments'), 0)),
        ))
    if hdr['MeasYaps'][('ucOneSeriesForAllMeas', )] in [2, 8]:
        user_parameter_long.append(ismrmrd.xsd.userParameterLongType(
            name='MultiSeriesForSlices',
            value=int(hdr['MeasYaps'][('ucOneSeriesForAllMeas', )]),
        ))
    if ('sPat', 'lRefLinesPE') in hdr['MeasYaps']:
        user_parameter_long.append(ismrmrd.xsd.userParameterLongType(
            name='EmbeddedRefLinesE1',
            value=int(hdr['MeasYaps'][('sPat', 'lRefLinesPE')]),
        ))
    if ('sPat', 'lRefLines3D') in hdr['MeasYaps']:
        user_parameter_long.append(ismrmrd.xsd.userParameterLongType(
            name='EmbeddedRefLinesE2',
            value=int(hdr['MeasYaps'][('sPat', 'lRefLines3D')]),
        ))
    if 'lProtonDensMap' in hdr['MeasYaps']:
        user_parameter_long.append(ismrmrd.xsd.userParameterLongType(
            name='NumOfProtonDensityImages',
            value=int(hdr['MeasYaps']['lProtonDensMap']),
        ))
    if 'relSliceNumber' in hdr['Config']:
        relSliceNumber = hdr['Config']['relSliceNumber'].split()
        relSliceNumber = [int(x) for x in relSliceNumber if int(x) >= 0]
        for i, slice_number in enumerate(relSliceNumber):
            user_parameter_long.append(ismrmrd.xsd.userParameterLongType(
                name=f'RelativeSliceNumber_{i+1}',
                value=slice_number,
            ))

    # userParameterDouble
    user_parameter_double = []
    if ('sWipMemBlock', 'adFree') in hdr['MeasYaps']:
        for i, ad_free_str in enumerate(re.split(' +', hdr['MeasYaps'][('sWipMemBlock', 'adFree')])):
            user_parameter_double.append(ismrmrd.xsd.userParameterDoubleType(
                name=f'sWipMemBlock.adFree[{i}]',
                value=float(ad_free_str),
            ))
    for i in range(6):
        if ('sPrepPulses', 'adT2PrepDuration', str(i)) in hdr['MeasYaps']:
            user_parameter_double.append(ismrmrd.xsd.userParameterDoubleType(
                name=f'T2PrepDuration_{i}',
                value=float(hdr['MeasYaps'][('sPrepPulses', 'adT2PrepDuration', str(i))]),
            ))
    if 'aflMaxwellCoefficients' in hdr['Meas']:
        for i, coeff_str in enumerate(re.split(' +', hdr['Meas']['aflMaxwellCoefficients'])):
            user_parameter_double.append(ismrmrd.xsd.userParameterDoubleType(
                name=f'MaxwellCoefficient_{i}',
                value=float(coeff_str),
            ))
            if i >= 15:
                break
    if ('sSliceArray', 'asSlice', '0', 'dInPlaneRot') in hdr['MeasYaps']:
        user_parameter_double.append(ismrmrd.xsd.userParameterDoubleType(
            name='InPlaneRot',
            value=hdr['MeasYaps'][('sSliceArray', 'asSlice', '0', 'dInPlaneRot')],
        ))
    if 'flContrastBolusVolume' in hdr['Meas'] and hdr['Meas']['flContrastBolusVolume'] != '':
        user_parameter_double.append(ismrmrd.xsd.userParameterDoubleType(
            name='ContrastBolusVolume',
            value=hdr['Meas']['flContrastBolusVolume'],
        ))
    if 'flContrastBolusTotalDose' in hdr['Meas'] and hdr['Meas']['flContrastBolusTotalDose'] != '':
        user_parameter_double.append(ismrmrd.xsd.userParameterDoubleType(
            name='ContrastBolusTotalDose',
            value=hdr['Meas']['flContrastBolusTotalDose'],
        ))

    # userParameterString
    user_parameter_string = []
    if 'tContrastBolusAgent' in hdr['Meas'] and hdr['Meas']['tContrastBolusAgent'] != '':
        user_parameter_string.append(ismrmrd.xsd.userParameterStringType(
            name='ContrastBolusAgent',
            value=hdr['Meas']['tContrastBolusAgent'],
        ))
    if hdr['MeasYaps'][('sPhysioImaging', 'lSignal1')] not in [1, 16] \
            and hdr['MeasYaps'][('sPhysioImaging', 'lMethod1')] == 8:
        if hdr['MeasYaps'][('sPhysioImaging', 'lSignal1')] == 2:
            user_parameter_string.append(ismrmrd.xsd.userParameterStringType(
                name='RetroGatingMode',
                value='ECG',
            ))
        if hdr['MeasYaps'][('sPhysioImaging', 'lSignal1')] == 4:
            user_parameter_string.append(ismrmrd.xsd.userParameterStringType(
                name='RetroGatingMode',
                value='External',
            ))

    user_parameters = ismrmrd.xsd.userParametersType(
        userParameterLong=user_parameter_long,
        userParameterDouble=user_parameter_double,
        userParameterString=user_parameter_string,
    )

    return user_parameters
