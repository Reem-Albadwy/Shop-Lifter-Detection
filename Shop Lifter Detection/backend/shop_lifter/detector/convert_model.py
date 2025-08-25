import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM

model = load_model(
    r"C:\Users\Huawei\OneDrive\Desktop\Shop Lifter Detection\Shop_Lifter_Tuned_Model_Modern.keras",
    compile=False
)

def make_lstm_cpu_friendly(layer):
    if isinstance(layer, tf.keras.layers.LSTM):
        config = layer.get_config()
        config['activation'] = 'tanh'
        config['recurrent_activation'] = 'sigmoid'
        config['use_bias'] = True
        config['unit_forget_bias'] = True
        return tf.keras.layers.LSTM.from_config(config)
    return layer

new_model = tf.keras.models.clone_model(
    model,
    clone_function=make_lstm_cpu_friendly
)

new_model.set_weights(model.get_weights())

new_model.save(
    r"C:\Users\Huawei\OneDrive\Desktop\Shop Lifter Detection\Shop_Lifter_Tuned_Model_CPU",
    save_format="tf"
)

print("✅ تم إنشاء موديل جديد متوافق مع CPU بنجاح")
