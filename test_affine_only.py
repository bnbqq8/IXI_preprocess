# test affine tranform only based on volume affine matrix
import time

import numpy as np
import SimpleITK as sitk

from utils import Registration, nmi, nmi_wrap

current_level_start_time = None
current_level = 0


def command_end(method):
    """Callback invoked at the end of the registration process"""
    global current_level_start_time, current_level
    print("Optimizer stop condition: " + method.GetOptimizerStopConditionDescription())
    print(f" Iteration: {method.GetOptimizerIteration()}")
    print(f" Metric value: {method.GetMetricValue()}")

    now = time.time()
    duration = now - current_level_start_time
    print(f"Resolution level {current_level} took {duration:.2f} seconds.")
    print("--------- Registration Finished ---------")


def command_multi_iteration(method):
    """Callback invoked before starting a multi-resolution level.
    The sitkMultiResolutionIterationEvent occurs before the
    resolution of the transform. This event is used here to print
    the status of the optimizer from the previous registration level.
    """
    global current_level_start_time, current_level
    now = time.time()
    if method.GetCurrentLevel() > 0:
        print(
            "Optimizer stop condition: "
            + f"{method.GetOptimizerStopConditionDescription()}"
        )
        print(f" Iteration: {method.GetOptimizerIteration()}")
        print(f" Metric value: {method.GetMetricValue()}")
        print(f" Custom metric value: {nmi_wrap(method)}")

        if current_level_start_time is not None:
            duration = now - current_level_start_time
            print(f"Resolution level {current_level} took {duration:.2f} seconds.")
            current_level += 1
        else:
            print("Starting registration multi-resolution process.")
    current_level_start_time = now
    print("--------- Resolution Changing ---------")


def get_index_to_point_matrix(image: sitk.Image) -> np.ndarray:
    # 获得图像的维度
    dim = image.GetDimension()

    # 提取 spacing 和 origin 信息
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())

    # 将 direction 重构成二维矩阵（维度: dim x dim）
    direction = np.array(image.GetDirection()).reshape(dim, dim)

    # 构造一个对角矩阵，包含 spacing
    spacing_mat = np.diag(spacing)

    # 计算 IndexToPointMatrix = direction * diag(spacing)
    index_to_point_matrix = direction @ spacing_mat

    # 构造 4x4 matrix，其中最后一列是 origin
    index_to_point_matrix_4x4 = np.eye(4)
    index_to_point_matrix_4x4[:dim, :dim] = index_to_point_matrix
    index_to_point_matrix_4x4[:dim, 3] = origin

    return index_to_point_matrix_4x4
    # return index_to_point_matrix_4x4[:3]


# moving:A fixed:B
def get_affine_tx(mat_a, mat_b):
    _mat = np.linalg.inv(mat_b) @ mat_a
    tx_mat = sitk.AffineTransform(3)
    tx_mat.SetMatrix(_mat[:3, :3].flatten())
    tx_mat.SetTranslation(_mat[:3, 3])
    return tx_mat


T1_path = "../IXI_dataset/IXI_registrated/IXI230-IOP-0869/T1.nii.gz"
T2_path = "../IXI_dataset/IXI_registrated/IXI230-IOP-0869/T2.nii.gz"

T1_image = sitk.ReadImage(str(T1_path), sitk.sitkFloat32)
T2_image = sitk.ReadImage(str(T2_path), sitk.sitkFloat32)

T1_affine_mat = get_index_to_point_matrix(T1_image)
T2_affine_mat = get_index_to_point_matrix(T2_image)

# np.set_printoptions(precision=4, suppress=True)
# print("T1 affine matrix:")
# print(T1_affine_mat)
# print("T2 affine matrix:")
# print(T2_affine_mat)

# # hook for command
# global _fixed_img, _moving_img
# _fixed_img = T2_image
# _moving_img = T1_image

affine_tx = get_affine_tx(T1_affine_mat, T2_affine_mat)
# print("Affine transform matrix:")
# print(affine_tx)

# normal affine transform

# out_T1 = sitk.Resample(
#     T1_image,
#     affine_tx,
#     sitk.sitkLinear,
#     0.0,
#     T1_image.GetPixelID(),
# )
# sitk.WriteImage(out_T1, "test.nii.gz")

# affine registration
# affine_reg = sitk.ImageRegistrationMethod()
# affine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
# affine_reg.SetShrinkFactorsPerLevel([32, 16, 8, 4, 2, 1])
# affine_reg.SetSmoothingSigmasPerLevel([32, 16, 8, 4, 2, 1])

# tx = sitk.AffineTransform(3)
# affine_reg.SetInitialTransform(tx)
# affine_reg.SetOptimizerAsRegularStepGradientDescent(

#     learningRate=1.0, minStep=1e-4, numberOfIterations=200
# )
# affine_reg.AddCommand(
#     sitk.sitkMultiResolutionIterationEvent, lambda: command_multi_iteration(affine_reg)
# )
# affine_reg.AddCommand(sitk.sitkEndEvent, lambda: command_end(affine_reg))

reg = Registration()

tmp, out = reg(T1_image, T2_image)

# total_start_time = time.time()

# affine_transform = affine_reg.Execute(T2_image, T1_image)

# print(f"Total time: {time.time() - total_start_time:.2f} seconds")

# print("Affine transform matrix from :")
# print(affine_transform)
# out_T1 = sitk.Resample(
#     T1_image,
#     T2_image,
#     affine_transform,
#     sitk.sitkLinear,
#     0.0,
#     T1_image.GetPixelID(),
# )
sitk.WriteImage(tmp, "test_affine_only.nii.gz")
sitk.WriteImage(out, "test_deformable.nii.gz")
