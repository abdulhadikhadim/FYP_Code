import tensorflow as tf

def categorical_dice_coef(y_true, y_pred, axis=(1, 2), smooth=1e-5):
    intersection = tf.reduce_sum(y_true * y_pred, axis=axis)
    union = tf.reduce_sum(y_true + y_pred, axis=axis)
    dice = (2. * intersection + smooth) / (union + smooth)
    return tf.reduce_mean(dice)

def categorical_dice_loss(y_true, y_pred):
    return 1.0 - categorical_dice_coef(y_true, y_pred)

