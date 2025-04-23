import os
from pathlib import Path

import ants
import pandas as pd
from tqdm import tqdm

from utils import nmi_ants

wightPDW = 1
metricParam = [32, "Regular", 0.2]  # bins, sampling, samplingPercentage for mattes

os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "60"
# T2 and PDW from resampled_dir, T1 from raw_dir
resampled_dir = Path("../IXI_dataset/IXI_registrated/")
raw_dir = Path("../IXI_dataset/IXI_raw/IXI-T1")

unalignments = []
log_df = pd.DataFrame(columns=["subject", "ants MI", "custom MI", "custom NMI"])
for patient_dir in tqdm(resampled_dir.iterdir()):
    print(patient_dir.name)

    fixed_image = ants.image_read(str(patient_dir / "T2.nii.gz"))
    fixed_image_PDW = ants.image_read(str(patient_dir / "PD.nii.gz"))
    try:
        moving_image = ants.image_read(str(raw_dir / f"{patient_dir.name}-T1.nii.gz"))
    except FileNotFoundError:
        print(f"{raw_dir / patient_dir.name}-T1.nii.gz not found")
        continue
    # # calc and print NMI between T2 & PDW, before n4 bias field correction
    # mi_ants, mi, nmi = nmi_ants(fixed_image, fixed_image_PDW)
    # print(f"T2 and PDW (before N4): ants{mi_ants}, {mi}, {nmi}")
    # N4 bias field correction (ANTs ver.)
    fixed_image = ants.n4_bias_field_correction(fixed_image)
    moving_image = ants.n4_bias_field_correction(moving_image)
    fixed_image_PDW = ants.n4_bias_field_correction(fixed_image_PDW)

    # rigid
    # mytx = ants.registration(
    #     fixed=fixed_image,
    #     moving=moving_image,
    #     aff_metric="mattes",
    #     syn_metric="mattes",
    #     type_of_transform="Rigid",
    #     initial_transform="identity",
    #     grad_step=0.1,
    #     aff_shrink_factors=(8, 4, 2, 1),
    #     aff_smoothing_sigmas=(4, 2, 1, 0),
    #     multivariate_extras=[
    #         "MattesMutualInformation",
    #         fixed_image_PDW,
    #         moving_image,
    #         wightPDW,
    #         metricParam,
    #     ],
    # )
    mytx_aff = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        aff_metric="mattes",
        syn_metric="mattes",
        type_of_transform="Affine",
        initial_transform="identity",
        grad_step=0.1,
        aff_iterations=(2100, 2100, 1200, 1200, 10),
        aff_shrink_factors=(16, 8, 4, 2, 1),
        aff_smoothing_sigmas=(8, 4, 2, 1, 0),
        multivariate_extras=[
            "MattesMutualInformation",
            fixed_image_PDW,
            moving_image,
            wightPDW,
            metricParam,
        ],
    )
    mytx_syn = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform="SyN",
        initial_transform=mytx_aff["fwdtransforms"],
        grad_step=0.1,
        syn_metric="mattes",
        multivariate_extras=[
            "MattesMutualInformation",
            fixed_image_PDW,
            moving_image,
            wightPDW,
            metricParam,
        ],
    )

    # calc and print NMI between T1 & T2 and T2 & PDW
    # mi_ants, mi, nmi = nmi_ants(fixed_image, fixed_image_PDW)
    # print(f"T2 and PDW  (after N4): ants{mi_ants}, {mi}, {nmi}")
    # mi_ants, mi, nmi = nmi_ants(fixed_image, moving_image)
    # print(f"T2 and T1        (raw): ants{mi_ants}, {mi}, {nmi}")
    # mi_ants, mi, nmi = nmi_ants(fixed_image, mytx["warpedmovout"])
    # print(f"T2 and T1        (rig): ants{mi_ants}, {mi}, {nmi}")
    # mi_ants, mi, nmi = nmi_ants(fixed_image, mytx_aff["warpedmovout"])
    # print(f"T2 and T1        (aff): ants{mi_ants}, {mi}, {nmi}")
    mi_ants, mi, nmi = nmi_ants(fixed_image, mytx_syn["warpedmovout"])
    # print(f"T2 and T1        (syn): ants{mi_ants}, {mi}, {nmi}")

    if nmi < 0.40:
        unalignments.append(patient_dir.name)
    # log results
    log_df = pd.concat(
        [
            log_df,
            pd.DataFrame(
                {
                    "subject": [patient_dir.name],
                    "ants MI": [mi_ants],
                    "custom MI": [mi],
                    "custom NMI": [nmi],
                }
            ),
        ],
        ignore_index=True,
    )
    # save to resampled_dir
    fixed_image.to_file(str(patient_dir / "T2_ants_N4.nii.gz"))
    fixed_image_PDW.to_file(str(patient_dir / "PD_ants_N4.nii.gz"))
    mytx_aff["warpedmovout"].to_file(
        str(patient_dir / "T1_ants_N4_registration.nii.gz")
    )
# print unalignments and save the list to csv
print("NMI < 0.4 after registration:")
print(*unalignments, sep="\n")
with open("unalignments(NMI<0.4).csv", "w") as f:
    for item in unalignments:
        f.write("%s\n" % item)
# save log
log_df.set_index("subject", inplace=True)
log_df.to_csv("ants_registration_log.csv")
