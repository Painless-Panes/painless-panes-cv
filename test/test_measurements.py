import cv2

import painless_panes

VIEW = True


def test__measurement():
    def test_(file_name, ref_width, ref_height, tolerance=1, enforce=True):
        print()
        print(f"Testing {file_name}...")

        image = cv2.imread(f"original/{file_name}")
        image = shrink_to_fit(image)
        (
            meas_width,
            meas_height,
            image_annotated,
            message,
        ) = painless_panes.cv.measure_window(image)

        print("Message:", message)
        print(f"Reference dimensions: {ref_width} x {ref_height}")
        print(f"Measured dimensions:  {meas_width} x {meas_height}")

        cv2.imwrite(f"results/{file_name}", image_annotated)
        if VIEW:
            cv2.imshow(file_name, image_annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # If enforcing the test, assert that the measurements are within tolerance
        if enforce:
            assert abs(meas_width - ref_width) <= tolerance
            assert abs(meas_height - ref_height) <= tolerance

    test_("example1.jpg", 26, 66, enforce=False)
    test_("example2.jpg", 28, 57, enforce=False)
    test_("example3.jpg", 26, 66, enforce=False)
    test_("example4.jpg", 26, 66, enforce=False)
    test_("example5.png", 28, 34, enforce=False)
    test_("example6.png", 28, 54, enforce=False)
    test_("example7.png", 28, 54, enforce=False)


def shrink_to_fit(image, height=800):
    image = image.copy()
    height0, width0, _ = image.shape

    width = int(width0 * height / height0)
    image = cv2.resize(image, (width, height))
    return image


test__measurement()
