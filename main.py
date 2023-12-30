import tensorflow as tf

dtype = tf.int8


class TestModel(tf.Module):
    def __init__(self):
        super(TestModel, self).__init__()

    @tf.lite.experimental.authoring.compatible
    @tf.function(input_signature=[tf.TensorSpec(shape=[1, 224, 224, 3], dtype=dtype)])
    def add(self, x):
        """
        Simple method that accepts single input 'x' and returns 'x' + 4.
        """
        # Name the output 'result' for convenience.
        return {'result': x + tf.constant([1], dtype=dtype)}


SAVED_MODEL_PATH = 'content/saved_models/test_variable'
TFLITE_FILE_PATH = 'content/test_variable.tflite'

# Save the model
module = TestModel()
# You can omit the signatures argument and a default signature name will be
# created with name 'serving_default'.
tf.saved_model.save(
    module, SAVED_MODEL_PATH, signatures={'my_signature': module.add.get_concrete_function()}
)

# Convert the model using TFLiteConverter
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
]
tflite_model = converter.convert()
with open(TFLITE_FILE_PATH, 'wb') as f:
    f.write(tflite_model)

# Load the TFLite model in TFLite Interpreter
interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
# There is only 1 signature defined in the model,
# so it will return it by default.
# If there are multiple signatures then we can pass the name.
my_signature = interpreter.get_signature_runner()

# my_signature is callable with input as arguments.
output = my_signature(x=tf.constant([1], shape=(1, 224, 224, 3), dtype=dtype))
# 'output' is dictionary with all outputs from the inference.
# In this case we have single output 'result'.
print(output['result'][:10], '...')
