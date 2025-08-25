import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force TensorFlow to use CPU

import cv2
import numpy as np
import tensorflow as tf
from django.shortcuts import render, redirect
from django.http import HttpResponse

def home(request):
    return redirect('predict_image')

# تحميل الموديل الجديد المتوافق مع CPU
model_path = r"C:\Users\Huawei\OneDrive\Desktop\Shop Lifter Detection\Shop_Lifter_Tuned_Model_tf"
model = tf.saved_model.load(model_path)
infer = model.signatures["serving_default"]

def predict_image(request):
    if request.method == 'POST' and request.FILES.get('video_file'):
        video_file = request.FILES['video_file']
        video_path = 'temp_video.mp4'

        # حفظ الفيديو مؤقتًا
        with open(video_path, 'wb+') as f:
            for chunk in video_file.chunks():
                f.write(chunk)

        # قراءة الفيديو واستخراج الإطارات
        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)

        cap.release()

        # تحويل الإطارات إلى NumPy array وتطبيع القيم
        frames = np.array(frames) / 255.0

        # التأكد من وجود 16 إطار بالظبط
        seq_len = 16
        if len(frames) < seq_len:
            frames = np.tile(frames, (seq_len // len(frames) + 1, 1, 1, 1))
        frames = frames[:seq_len]

        # إضافة بُعد batch وتحويلها إلى Tensor
        frames_tensor = tf.convert_to_tensor(np.expand_dims(frames, axis=0), dtype=tf.float32)

        # التنبؤ بالموديل
        prediction = infer(keras_tensor_154=frames_tensor)  # الاسم ممكن يختلف حسب signature
        pred_value = list(prediction.values())[0].numpy()[0][0]

        result = "Thief" if pred_value > 0.5 else "Not Thief"
        return HttpResponse(f"Prediction: {result}")

    return render(request, 'upload.html')
