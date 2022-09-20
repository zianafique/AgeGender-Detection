import cv2
import dlib
import numpy as np
import urllib.request as ur
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from openvino.runtime import Core


def model_init(model):
    ie_core = Core()
    model = ie_core.read_model(model=model)
    compiled_model = ie_core.compile_model(model=model, device_name="CPU")
    input_keys = compiled_model.input(0)
    output_keys = compiled_model.output(0)
    return input_keys, output_keys, compiled_model


def AgeGender_detection(image_encoded):
    ir_model_xml = "saved_model.xml"

    inputs, outputs, compiled_model = model_init(ir_model_xml)
    margin = 0.4
    img_size = 64
    detector = dlib.get_frontal_face_detector()
    decoded = ur.urlopen(image_encoded)
    img = decoded.file.read()
    np_data = np.frombuffer(img, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)
    img_h, img_w, _ = np.shape(img)
    detected = detector(img, 1)
    faces = np.empty((len(detected), img_size, img_size, 3))

    if len(detected) > 0:
        for i, d in enumerate(detected):
            x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + \
                1, d.bottom() + 1, d.width(), d.height()
            xw1 = max(int(x1 - margin * w), 0)
            yw1 = max(int(y1 - margin * h), 0)
            xw2 = min(int(x2 + margin * w), img_w - 1)
            yw2 = min(int(y2 + margin * h), img_h - 1)
            faces[i, :, :, :] = cv2.resize(
                img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

        # make a prediction for Age and Gender
        results = compiled_model([faces])

        a = dict(zip(['gender', 'age'], results.values()))
        age = a['age']
        gender = a['gender']
        ages = np.arange(0, 101).reshape(101, 1)
        predicted_ages = age.dot(ages).flatten()
        Gender="Female" if gender[0][0] > 0.4 else "Male"

        with open("results.txt", 'w') as outfile:
            outfile.write("The Face is of a person who is :"+ Gender+ '\n')
            outfile.write("The age of that person is: %d\n" % (int(predicted_ages)))
        return((int(predicted_ages)), "Female" if gender[0][0] > 0.4 else "Male")


selfie_file = open("Kai.txt", "r")
selfie = selfie_file.read()

print(AgeGender_detection(selfie))
