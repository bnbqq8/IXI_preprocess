import SimpleITK as sitk


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
    if method.GetCurrentLevel() > 0:
        print(
            "Optimizer stop condition: "
            + f"{method.GetOptimizerStopConditionDescription()}"
        )
        print(f" Iteration: {method.GetOptimizerIteration()}")
        print(f" Metric value: {method.GetMetricValue()}")

    print("--------- Resolution Changing ---------")


class Registration:
    def __init__(self):
        self.affine_reg_init()
        self.bspline_reg_init()

    def affine_reg_init(self):
        self.affine_reg = sitk.ImageRegistrationMethod()
        self.affine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        self.affine_reg.SetShrinkFactorsPerLevel([8, 4])
        self.affine_reg.SetSmoothingSigmasPerLevel([8, 4])
        # self.affine_reg.AddCommand(
        #     sitk.sitkIterationEvent, lambda: command_iteration(self.affine_reg)
        # )
        # self.affine_reg.AddCommand(
        #     sitk.sitkMultiResolutionIterationEvent,
        #     lambda: command_multi_iteration(self.affine_reg),
        # )

    def bspline_reg_init(self):
        # B-Spline transform
        self.bspline_reg = sitk.ImageRegistrationMethod()
        self.bspline_reg.SetMetricAsJointHistogramMutualInformation()

        self.bspline_reg.SetInterpolator(sitk.sitkLinear)

        self.bspline_reg.SetShrinkFactorsPerLevel([4, 2, 1])
        self.bspline_reg.SetSmoothingSigmasPerLevel([4, 2, 1])
        # self.bspline_reg.AddCommand(
        #     sitk.sitkIterationEvent, lambda: command_iteration(self.bspline_reg)
        # )
        # self.bspline_reg.AddCommand(
        #     sitk.sitkMultiResolutionIterationEvent,
        #     lambda: command_multi_iteration(self.bspline_reg),
        # )

    def affine_tx_init(self):
        tx = sitk.AffineTransform(3)
        self.affine_reg.SetInitialTransform(tx)
        self.affine_reg.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=1e-4, numberOfIterations=200
        )

    def bspline_tx_init(self, fixed_img):
        transformDomainMeshSize = [2] * 3
        tx = sitk.BSplineTransformInitializer(fixed_img, transformDomainMeshSize)
        self.bspline_reg.SetInitialTransformAsBSpline(
            tx, inPlace=True, scaleFactors=[1, 2, 5]
        )
        self.bspline_reg.SetOptimizerAsGradientDescentLineSearch(
            5.0, 100, convergenceMinimumValue=1e-4, convergenceWindowSize=5
        )

    def __call__(self, moving_img: sitk.Image, fixed_img: sitk.Image):

        # get affine transform
        self.affine_tx_init()
        affine_transform = self.affine_reg.Execute(fixed_img, moving_img)

        # get affine transformed moving image
        moving_resampled_affine = sitk.Resample(
            moving_img,
            fixed_img,
            affine_transform,
            sitk.sitkLinear,
            0.0,
            moving_img.GetPixelID(),
        )
        # get elastic transform
        self.bspline_tx_init(fixed_img)
        elastic_transform = self.bspline_reg.Execute(fixed_img, moving_resampled_affine)

        # combine transforms and return
        # composite_transform = sitk.Transform(3, sitk.sitkComposite)
        # composite_transform.AddTransform(affine_transform)
        # composite_transform.AddTransform(elastic_transform)
        composite_transform = sitk.CompositeTransform(
            [affine_transform, elastic_transform]
        )
        return moving_resampled_affine, sitk.Resample(
            moving_img,
            fixed_img,
            composite_transform,
            sitk.sitkLinear,
            0.0,
            moving_img.GetPixelID(),
        )
