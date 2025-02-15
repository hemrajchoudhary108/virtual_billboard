import cv2
import numpy as np

# Defining window name
window_name = 'Image'

# ---------------------------------------------------------------------
# Function to display the image
# ---------------------------------------------------------------------

def mouse_handler(event, x, y, flags, data) -> None:
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(data['im'], (x, y), 10, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.imshow("Image", data['im'])
        if len(data['points']) < 4:
            data['points'].append([x, y])

# ----------------------------------------------------------------------
# Function to get the ROI points
# ----------------------------------------------------------------------
def get_roi_points(image: np.ndarray) -> np.ndarray:
    # Creating data storage dictionary
    data = {}
    data['im'] = image.copy()
    data['points'] = []

    # Display the image
    cv2.imshow(window_name, data['im'])

    # Setting mouse handler for the image
    cv2.setMouseCallback(window_name, mouse_handler, data)
    cv2.waitKey(0)

    # Storing points
    points = np.array(data['points'])

    return points

# ----------------------------------------------------------------------
# Main processing function
# ----------------------------------------------------------------------

# Reading the image
image_src_path = 'destination_image.png'
image = cv2.imread(image_src_path)

# Reading the destination image
dst_img_path = 'dst.jpg'
dst_image = cv2.imread(dst_img_path)

# Calculating the cordinates of four corners of src image
src_h, src_w, _ = image.shape
src_points = np.array([[0, 0], [src_w - 1, 0], [src_w-1, src_h-1], [0, src_h-1]], dtype=np.float32)

print('Select the ROI points in the destination image, Note: select in the order of top-left, top-right, bottom-right, bottom-left')
dst_points = get_roi_points(dst_image)

# Calculating the homography matrix
h, status = cv2.findHomography(src_points, dst_points)

# Black out the destination image of selected ROI
cv2.fillConvexPoly(dst_image, dst_points.astype(int), 0, 16)

# Warp the source image to destination based on homography
warped_image = cv2.warpPerspective(image, h, (dst_image.shape[1], dst_image.shape[0]))

# Add the warped image to destination image
final_image = dst_image + warped_image

# Display the final image
cv2.imshow('Final Image', final_image)
cv2.waitKey(0)
cv2.imwrite('Final.png', final_image)