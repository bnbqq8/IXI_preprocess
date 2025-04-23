# Exclusion + N4 bias field correction + Resampling + Refactor directory structure
from pathlib import Path

import SimpleITK as sitk
from tqdm import tqdm

from utils import Registration, command_iteration, command_multi_iteration

registrator = Registration()
# for itk
# registrator = Registration_itk()

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
root_dir = Path("../IXI_dataset/IXI_registrated")
tar_dir = Path("../IXI_dataset/IXI_registrated")
seqs = ["T2", "T1"]


for subject in (pbar := tqdm(sorted((root_dir).iterdir(), key=lambda x: x.name)[:5])):
    pbar.set_postfix_str(subject.name)
    for seq in seqs:

        img_path = subject / f"{seq}.nii.gz"
        assert img_path.exists(), f"{img_path} does not exist"
        # for sitk
        _img = sitk.ReadImage(str(img_path), sitk.sitkFloat32)
        # for itk
        # _img = itk.imread(str(img_path), itk.F)

        # hook T1 and T2 for registration
        if seq == "T1":
            moving_img = _img
        if seq == "T2":
            fixed_img = _img
    # registration
    pbar.set_description_str(f"Registration")
    tmp, registered_img = registrator(moving_img, fixed_img)
    sitk.WriteImage(tmp, str(subject / f"T1_affine_tmp.nii.gz"))
    sitk.WriteImage(registered_img, str(subject / f"T1_bspline.nii.gz"))
