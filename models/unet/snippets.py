#U-Net snippets (selected fragments only).
#Full training and data pipeline withheld; available upon request.

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate, BatchNormalization, Dropout, Input
from tensorflow.keras import Model

def conv_block(x, f: int):
    x = Conv2D(f, 3, padding="same", activation="relu")(x); x = BatchNormalization()(x)
    x = Conv2D(f, 3, padding="same", activation="relu")(x); x = BatchNormalization()(x)
    return x

def down_block(x, f: int):
    c = conv_block(x, f)
    p = MaxPooling2D()(c)
    return c, p

def up_block(x, skip, f: int, drop: float | None = None):
    x = Conv2DTranspose(f, 2, strides=2, padding="same")(x)
    x = Concatenate()([x, skip])
    x = conv_block(x, f)
    if drop:
        x = Dropout(drop)(x)
    return x


def build_unet_skeleton(input_shape=(256, 256, 1), base_filters=64, depth=5, drops=(0.3, 0.2, 0.2, 0.1)):
    inputs = Input(input_shape)

    # Encoder (3 downs)
    c1 = conv_block(inputs, base_filters);       p1 = MaxPooling2D()(c1)
    c2 = conv_block(p1, base_filters * 2);       p2 = MaxPooling2D()(c2)
    c3 = conv_block(p2, base_filters * 4);       p3 = MaxPooling2D()(c3)

    # Bottleneck
    bn = conv_block(p3, base_filters * 8)

    # Decoder (3 ups)
    u3 = Conv2DTranspose(base_filters * 4, 2, strides=2, padding="same")(bn)
    u3 = Concatenate()([u3, c3])
    c4 = conv_block(u3, base_filters * 4)

    u2 = Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same")(c4)
    u2 = Concatenate()([u2, c2])
    c5 = conv_block(u2, base_filters * 2)

    u1 = Conv2DTranspose(base_filters, 2, strides=2, padding="same")(c5)
    u1 = Concatenate()([u1, c1])
    c6 = conv_block(u1, base_filters)

    # Output
    outputs = Conv2D(1, 1, activation="sigmoid")(c6)

    return Model(inputs, outputs, name="U-Net_skeleton")
