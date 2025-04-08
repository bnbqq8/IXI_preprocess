# Exclusion + N4 bias field correction + Resampling + Refactor directory structure
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm

from utils import Registration, command_iteration, command_multi_iteration


def n4_bias_field_correction(img_path: Path):
    img = sitk.ReadImage(str(img_path), sitk.sitkFloat32)
    mask = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    return corrector.Execute(img, mask)


def resample(img: sitk.Image):
    original_size = img.GetSize()
    original_spacing = img.GetSpacing()
    new_spacing = (original_spacing[0], original_spacing[1], original_spacing[0])
    new_size = [
        int(round(osz * ospc / nspc))
        for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkBSpline)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    return resample.Execute(img)


registrator = Registration()

# Spacing for T2 subject ../IXI_dataset/IXI_raw/IXI-T2/IXI014-HH-1236-T2.nii.gz: (0.8984375204972158, 4.977778057095752)
# T1-T2: {'IXI116-Guys-0739-', 'IXI182-Guys-0792-', 'IXI309-IOP-0897-', 'IXI500-Guys-1017-'}
# T2-T1: {'IXI580-IOP-1157-'}
exclusion = [
    "IXI116-Guys-0739",
    "IXI182-Guys-0792",
    "IXI309-IOP-0897",
    "IXI500-Guys-1017",
    "IXI580-IOP-1157",
    "IXI014-HH-1236",
]
root_dir = Path("../IXI_dataset/IXI_raw")
tar_dir = Path("../IXI_dataset/IXI_registrated")
seqs = ["T2", "PD", "T1"]


for subject in (
    pbar := tqdm(sorted((root_dir / "IXI-T2").iterdir(), key=lambda x: x.name)[200:])
):
    pbar.set_description_str(f"Resampling")
    pbar.set_postfix_str(subject.name)
    subject_prefix = subject.name.split("-T2.nii.gz")[0]
    if subject_prefix in exclusion:
        continue

    # refactor directory structure as following:
    # IXI_registrated
    #    |---subject1
    #    |    |---T2.nii.gz
    #    |    |---PD.nii.gz
    #    |    |---T1.nii.gz
    #    |---subject2
    #    |---subject3
    subject_tar_dir = tar_dir / subject_prefix
    subject_tar_dir.mkdir(exist_ok=True, parents=True)
    for seq in seqs:

        img_path = root_dir / f"IXI-{seq}/{subject_prefix}-{seq}.nii.gz"
        assert img_path.exists(), f"{img_path} does not exist"
        # check existence
        tar_path = subject_tar_dir / f"{seq}.nii.gz"
        if tar_path.exists():
            print(f"{tar_path} already exists")
            _img = sitk.ReadImage(str(tar_path), sitk.sitkFloat32)
        else:
            # N4 BFC
            _img = n4_bias_field_correction(img_path)
            # resample
            # if seq in ["T2", "PD"]:
            _img = resample(_img)
            sitk.WriteImage(_img, str(subject_tar_dir / f"{seq}.nii.gz"))

        # hook T1 and T2 for registration
        if seq == "T1":
            moving_img = _img
        if seq == "T2":
            fixed_img = _img
    # registration
    pbar.set_description_str(f"Registration")
    tmp, registered_img = registrator(moving_img, fixed_img)
    sitk.WriteImage(tmp, str(subject_tar_dir / f"T1_affine_tmp.nii.gz"))
    sitk.WriteImage(registered_img, str(subject_tar_dir / f"T1_bspline.nii.gz"))
