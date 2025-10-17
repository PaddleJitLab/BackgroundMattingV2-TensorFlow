import tensorflow as tf

from model import MattingRefine

model = MattingRefine(backbone='resnet50',
                      backbone_scale=0.25,
                      refine_mode='sampling',
                      refine_sample_pixels=80000)

try:
    # for keras
    tf.saved_model.save(model, "./export_model")

    # for tf.Model
    # tf.saved_model.save(
    #     model,
    #     export_dir="./export_model",
    #     signatures=[
    #         tf.TensorSpec(
    #             shape=(1, 1080, 1920, 3), dtype=tf.float32, name="input_image"
    #         ),
    #         tf.TensorSpec(
    #             shape=(1, 1080, 1920, 3), dtype=tf.float32, name="input_bg"
    #         ),
    #     ],
    # )
    print("[JIT] Export model by TensorFlow successed.")
except:
    print("[JIT] Export model by TensorFlow failed.")
