import time

import ants
import itk
import itk.itkNormalizedCorrelationImageToImageMetricPython
import numpy as np
import SimpleITK as sitk
from cv2 import transform


def command_iteration(method):
    """Callback invoked each iteration"""
    if method.GetOptimizerIteration() == 0:
        # The BSpline is resized before the first optimizer
        # iteration is completed per level. Print the transform object
        # to show the adapted BSpline transform.
        print(method.GetInitialTransform())

    print(f"{method.GetOptimizerIteration():3} " + f"= {method.GetMetricValue():10.5f}")


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
        # test in-place
        # print(method.GetInitialTransform())

        if current_level_start_time is not None:
            duration = now - current_level_start_time
            print(f"Resolution level {current_level} took {duration:.2f} seconds.")
            current_level += 1
        else:
            print("Starting registration multi-resolution process.")
    current_level_start_time = now
    print("--------- Resolution Changing ---------")


def command_end(method):
    """Callback invoked at the end of the registration process"""
    global current_level_start_time, current_level
    print("Optimizer stop condition: " + method.GetOptimizerStopConditionDescription())
    print(f" Iteration: {method.GetOptimizerIteration()}")
    print(f" Metric value: {method.GetMetricValue()}")
    print(f" Custom metric value: {nmi_wrap(method)}")
    now = time.time()
    duration = now - current_level_start_time
    print(f"Resolution level {current_level} took {duration:.2f} seconds.")
    print("--------- Registration Finished ---------")


# def affine_moving_to_fixed(moving_img, fixed_img):


class Registration:
    def __init__(self):
        self.affine_reg_init()
        self.bspline_reg_init()

    def affine_reg_init(self):
        self.affine_reg = sitk.ImageRegistrationMethod()
        print(
            f"Affine Registration get inplace:{self.affine_reg.GetInitialTransformInPlace()}"
        )
        self.affine_reg.SetInterpolator(sitk.sitkLinear)
        self.affine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        self.affine_reg.SetShrinkFactorsPerLevel([32, 16, 8, 4, 2, 1])
        self.affine_reg.SetSmoothingSigmasPerLevel([32, 16, 8, 4, 2, 1])
        self.affine_reg.SetOptimizerScalesFromPhysicalShift()
        # self.affine_reg.AddCommand(
        #     sitk.sitkIterationEvent, lambda: command_iteration(self.affine_reg)
        # )
        self.affine_reg.AddCommand(
            sitk.sitkMultiResolutionIterationEvent,
            lambda: command_multi_iteration(self.affine_reg),
        )
        self.affine_reg.AddCommand(
            sitk.sitkEndEvent, lambda: command_end(self.affine_reg)
        )

    def bspline_reg_init(self):
        # B-Spline transform
        self.bspline_reg = sitk.ImageRegistrationMethod()

        self.bspline_reg.SetInterpolator(sitk.sitkLinear)
        self.bspline_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        self.bspline_reg.SetMetricSamplingStrategy(
            sitk.ImageRegistrationMethod.RANDOM,
        )
        self.bspline_reg.SetMetricSamplingPercentage(0.01)
        # self.bspline_reg.SetMetricSamplingPercentagePerLevel(
        #     [0.01, 0.01, 0.01], sitk.sitkWallClock
        # )

        self.bspline_reg.SetShrinkFactorsPerLevel([4, 2])
        self.bspline_reg.SetSmoothingSigmasPerLevel([2, 1])
        self.bspline_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        # self.affine_reg.SetOptimizerScalesFromPhysicalShift()
        # self.bspline_reg.AddCommand(
        #     sitk.sitkIterationEvent, lambda: command_iteration(self.bspline_reg)
        # )
        self.bspline_reg.AddCommand(
            sitk.sitkMultiResolutionIterationEvent,
            lambda: command_multi_iteration(self.bspline_reg),
        )
        self.bspline_reg.AddCommand(
            sitk.sitkEndEvent, lambda: command_end(self.affine_reg)
        )

    def affine_tx_init(self):
        # tx = sitk.AffineTransform(3)
        tx = sitk.Euler3DTransform()
        self.affine_reg.SetInitialTransform(tx)
        self.affine_reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-4, numberOfIterations=200
        )
        # reset timer
        global current_level_start_time, current_level
        current_level_start_time = None
        current_level = 0

    def bspline_tx_init(self, fixed_img, affine_tx):
        transformDomainMeshSize = [8] * 3
        tx = sitk.BSplineTransformInitializer(fixed_img, transformDomainMeshSize)

        self.bspline_reg.SetMovingInitialTransform(affine_tx)
        self.bspline_reg.SetInitialTransformAsBSpline(
            tx, inPlace=True, scaleFactors=[1, 2, 5]
        )
        self.bspline_reg.SetOptimizerAsLBFGS2(
            solutionAccuracy=1e-2,
            numberOfIterations=100,
            deltaConvergenceTolerance=0.01,
        )
        # self.bspline_reg.SetOptimizerAsGradientDescentLineSearch(
        #     5.0, 100, convergenceMinimumValue=1e-4, convergenceWindowSize=5
        # )

        # reset timer
        global current_level_start_time, current_level
        current_level_start_time = None
        current_level = 0

    def __call__(self, moving_img: sitk.Image, fixed_img: sitk.Image):
        # # 1. resample to target coordinate
        # moving_img = sitk.Resample(
        #     moving_img,
        #     fixed_img,
        #     sitk.Transform(3, sitk.sitkIdentity),
        #     sitk.sitkLinear,
        #     0.0,
        #     moving_img.GetPixelID(),
        # )

        # hook for command
        global _fixed_img, _moving_img
        _fixed_img = fixed_img
        _moving_img = moving_img

        # 2. get affine transform
        self.affine_tx_init()
        affine_transform = self.affine_reg.Execute(fixed_img, moving_img)

        # 3. get affine transformed moving image
        moving_resampled_affine = sitk.Resample(
            moving_img,
            fixed_img,
            affine_transform,
            sitk.sitkLinear,
            0.0,
            moving_img.GetPixelID(),
        )

        # 4. get elastic transform
        self.bspline_tx_init(fixed_img, affine_transform)
        elastic_transform = self.bspline_reg.Execute(fixed_img, moving_img)

        # combine transforms and return
        # composite_transform = sitk.Transform(3, sitk.sitkComposite)
        # composite_transform.AddTransform(affine_transform)
        # composite_transform.AddTransform(elastic_transform)
        # composite_transform = sitk.CompositeTransform(
        #     [affine_transform, elastic_transform]
        # )
        composite_transform = sitk.CompositeTransform(3)
        composite_transform.AddTransform(affine_transform)
        composite_transform.AddTransform(elastic_transform)
        return moving_resampled_affine, sitk.Resample(
            moving_img,
            fixed_img,
            composite_transform,
            sitk.sitkLinear,
            0.0,
            moving_img.GetPixelID(),
        )


class Registration_itk:
    def __init__(self):
        self._type = itk.itkImagePython.itkImageF3

        # type definitions
        dim = 3
        coordinateRepType = 3
        self.rigidType = itk.AffineTransform[itk.D, dim]
        self.deformableType = itk.BSplineTransform[coordinateRepType, dim, 3]

        self.metricType = itk.itkNormalizedMutualInformationHistogramImageToImageMetric[
            self._type, self.type
        ]

        self.affine_reg_init()
        self.bspline_reg_init()

    def affine_reg_init(self):
        self.affine_reg = itk.ImageRegistrationMethod[self._type, self._type].New()

        self.affine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        self.affine_reg.SetShrinkFactorsPerLevel([32, 16, 8, 4, 2, 1])
        self.affine_reg.SetSmoothingSigmasPerLevel([32, 16, 8, 4, 2, 1])


def nmi(moving_image, fixed_image, nbins=64):
    moving_arr = sitk.GetArrayFromImage(moving_image)
    fixed_arr = sitk.GetArrayFromImage(fixed_image)

    slice_min = min(moving_image.GetSize()[-1], fixed_image.GetSize()[-1])

    moving_arr = moving_arr[:slice_min, :, :].ravel()
    fixed_arr = fixed_arr[:slice_min, :, :].ravel()

    joint_hist, moving_edges, fixed_edges = np.histogram2d(
        moving_arr, fixed_arr, bins=nbins
    )
    joint_prob = joint_hist / np.sum(joint_hist)
    joint_entropy = -np.sum(joint_prob * np.log(joint_prob + 1e-10))

    fixed_prob = np.sum(joint_prob, axis=0)
    fixed_entropy = -np.sum(fixed_prob * np.log(fixed_prob + 1e-10))

    moving_prob = np.sum(joint_prob, axis=1)
    moving_entropy = -np.sum(moving_prob * np.log(moving_prob + 1e-10))

    mutual_info = fixed_entropy + moving_entropy - joint_entropy
    nmi = mutual_info / np.sqrt(fixed_entropy * moving_entropy)
    return mutual_info, nmi


def nmi_wrap(method: sitk.ImageRegistrationMethod):
    global _fixed_img, _moving_img
    transform = method.GetInitialTransform()
    # Apply the transform to the moving image
    moving_image_transformed = sitk.Resample(
        _moving_img,
        _fixed_img,
        transform,
        sitk.sitkLinear,
        0.0,
        _moving_img.GetPixelID(),
    )
    return nmi(moving_image_transformed, _fixed_img)


def nmi_ants(fixed_image, moving_image, nbins=32):
    mi_ants = ants.image_mutual_information(fixed_image, moving_image)
    fixed_arr = fixed_image.numpy()
    moving_arr = moving_image.numpy()

    slice_min = min(moving_arr.shape[-1], fixed_arr.shape[-1])

    moving_arr = moving_arr[:, :, :slice_min].ravel()
    fixed_arr = fixed_arr[:, :, :slice_min].ravel()

    joint_hist, moving_edges, fixed_edges = np.histogram2d(
        moving_arr, fixed_arr, bins=nbins
    )
    joint_prob = joint_hist / np.sum(joint_hist)
    joint_entropy = -np.sum(joint_prob * np.log(joint_prob + 1e-10))

    fixed_prob = np.sum(joint_prob, axis=0)
    fixed_entropy = -np.sum(fixed_prob * np.log(fixed_prob + 1e-10))

    moving_prob = np.sum(joint_prob, axis=1)
    moving_entropy = -np.sum(moving_prob * np.log(moving_prob + 1e-10))

    mi = fixed_entropy + moving_entropy - joint_entropy
    nmi = mi / np.sqrt(fixed_entropy * moving_entropy)
    return mi_ants, mi, nmi


# class Registration_v2:
#     def __init__(self):
#         self.R =

#     def __call__(self, ):
#         R = sitk.ImageRegistrationMethod()

#         pass
