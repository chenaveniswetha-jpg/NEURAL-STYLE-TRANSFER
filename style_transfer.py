import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (256, 256))
    img = img[tf.newaxis, :]
    return img

content_image = load_image("content.jpg.png")
style_image = load_image("style.jpg.png")
model = hub.load(
'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
)
stylized_image = model(
tf.constant(content_image),
tf.constant(style_image)
)[0]
plt.imshow(stylized_image[0])
plt.axis('off')
plt.show()
