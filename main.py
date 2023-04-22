import os
import cv2
import numpy as np
import re

def create_pyramid(image_p, level_n, orientation_n):
    # Create a Gaussian pyramid
    pyramid = []
    temp = image_p.copy()
    for i in range(level_n):
        temp = cv2.pyrDown(temp)
        pyramid.append(temp)

    features_pyramid = []

    for level in pyramid:
        rows, cols = level.shape[:2]
        level_features = []
        for i in range(orientation_n):
            angle = i * (360 / orientation_n)
            m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            # Calculate the new size of the output image
            cos = np.abs(m[0, 0])
            sin = np.abs(m[0, 1])
            new_cols = int((rows * sin) + (cols * cos))
            new_rows = int((rows * cos) + (cols * sin))
            # Adjust the translation part of the matrix to ensure the entire image is visible
            m[0, 2] += (new_cols / 2) - (cols / 2)
            m[1, 2] += (new_rows / 2) - (rows / 2)
            # Apply the rotation
            rotated = cv2.warpAffine(level, m, (new_cols, new_rows))
            # Add the rotated image to the level features list

            # Create a mask of the black pixels
            mask = cv2.inRange(rotated, (0, 0, 0), (0, 0, 0))
            rotated[mask > 0] = [255, 255, 255]

            kernel = np.ones((3, 3), np.uint8)
            rotated = cv2.dilate(rotated, kernel, iterations=1)
            level_features.append(rotated)

        # Add the level features list to the features pyramid list
        features_pyramid.append(level_features)
    return features_pyramid


def show_pyramid(pyramid):
    for level in pyramid:
        for img in level:
            cv2.imshow('Result', img)
            cv2.waitKey(0)


def template_matcher(main_image, images):
    matches = []
    for image_ in images:
        print("matching: ", image_[1])
        pyramid = create_pyramid(image_[0], 5, 8)
        for level in pyramid:
            for orientation in level:
                result = cv2.matchTemplate(main_image, orientation, cv2.TM_CCOEFF_NORMED)
                matches.append((result, image_[1]))
    return matches


def non_maximum_suppression(current_matches, threshold):
    filtered_matches_l = []
    for i, match in enumerate(current_matches):
        # Apply threshold
        loc = np.where(match[0] >= threshold)
        for pt in zip(*loc[::-1]):
            # Check if this point is already covered by another match
            covered = False
            for other_match in filtered_matches_l:
                if other_match[0][0] <= pt[0] <= other_match[1][0] and other_match[0][1] <= pt[1] <= other_match[1][1]:
                    covered = True
                    break
            if not covered:
                # Add bounding box to filtered matches
                filtered_matches_l.append(((pt[0], pt[1]), (pt[0] + 60, pt[1] + 60), match[1]))
    return filtered_matches_l


def read_training_images(directory, label_regex):
    images_l = []
    for filename in os.listdir(directory):
        match = re.match(label_regex, filename)
        if re.match(regex, filename):
            label = match.group(1)
            image_ = cv2.imread(os.path.join(directory, filename))
            images_l.append((image_, label))
        else:
            print("incorrect filename, probably read an incorrect image that was not in the training set")
    return images_l


def read_test_images(directory):
    test_images_l = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename))
        test_images_l.append(img)
    return test_images_l


def display(image, filtered_matches):
    for matched_image, bbox, class_ in filtered_matches:
        print(class_)
        cv2.rectangle(image, matched_image, bbox, (255, 0, 0), 2)
        cv2.putText(image, class_, (matched_image[0], matched_image[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0),
                    thickness=1)
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def template_match(test_image, training_images_p, threshold):
    matches = template_matcher(test_image, training_images_p)
    filtered = non_maximum_suppression(matches, threshold)
    display(test_image, filtered)


if __name__ == '__main__':
    regex = r'\d{3}-(\w+(?:-\w+)*)\.png'

    training_images_dir = './dataset_2/training_images'

    test_images_rotations_dir = './dataset_2/test_rotations/images/'
    test_images_rotations_ann_dir = './dataset_2/test_rotations/annotations'

    test_images_no_rotations_dir = './dataset_2/test_no_rotations/images/'
    test_images_no_rotations_ann_dir = './dataset_2/test_no_rotations/annotations'

    training_images = read_training_images(training_images_dir, regex)
    test_images = read_test_images(test_images_rotations_dir)

    for image_t in test_images:
        template_match(image_t, training_images, 0.80)
