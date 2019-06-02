import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer


class MyOptimizer(optimizer.Optimizer):

    def _apply_sparse(self, grad, var):
        grad_t = self._zeros_slot(var, "g", self._name)
        grad_t = state_ops.assign(grad_t, tf.zeros_like(grad_t), use_locking=self._use_locking)
        grad_t = state_ops.scatter_add(grad_t, grad.indices, grad.values, use_locking=self._use_locking)
        return self._apply_dense(grad_t, var)


class Grad(MyOptimizer):

    def __init__(self, learning_rate=0.001, use_locking=False, name="myGrad"):
        super(Grad, self).__init__(use_locking, name)
        self._lr = learning_rate

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr, var.dtype.base_dtype)
        var_update = state_ops.assign_sub(var, lr_t * grad)
        return var_update


class Mom(MyOptimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, use_locking=False, name="myMom"):
        super(Mom, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1

        self._beta1_power = None

    def _create_slots(self, var_list):
        self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", "m")

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1, var.dtype.base_dtype)
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta1_fix = 1.0 - beta1_power

        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix), use_locking=self._use_locking)
        return var_update

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1], name=name_scope)


class AdaMax(MyOptimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, use_locking=False, name="AdaMax"):
        super(AdaMax, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._beta1_power = None

    def _create_slots(self, var_list):
        self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", "m")
            self._zeros_slot(v, "v", "v")

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon, var.dtype.base_dtype)

        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta1_fix = 1 - beta1_power

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_t = state_ops.assign(m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking)
        v_t = state_ops.assign(v, tf.maximum(beta2_t * v, tf.abs(grad)), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix) / (v_t + epsilon_t), use_locking=self._use_locking)
        return var_update

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1], name=name_scope)


class Adam(MyOptimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, use_locking=False, name="myAdam"):
        super(Adam, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._beta1_power = None
        self._beta2_power = None

    def _create_slots(self, var_list):
        self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)
        self._beta2_power = tf.Variable(self._beta2, name="beta2_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", "m")
            self._zeros_slot(v, "v", "v")

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon, var.dtype.base_dtype)

        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)

        beta1_fix = 1 - beta1_power
        beta2_fix = 1 - beta2_power

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_t = state_ops.assign(m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking)
        v_t = state_ops.assign(v, beta2_t * v + (grad * grad) * (1 - beta2_t), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix) / (math_ops.sqrt(v_t / beta2_fix) + epsilon_t), use_locking=self._use_locking)
        return var_update

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1, use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(self._beta2_power * self._beta2, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2], name=name_scope)


class AMSGrad(MyOptimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, use_locking=False, name="myAMSGrad"):
        super(AMSGrad, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._beta1_power = None
        self._beta2_power = None

    def _create_slots(self, var_list):
        self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)
        self._beta2_power = tf.Variable(self._beta2, name="beta2_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", "m")
            self._zeros_slot(v, "v", "v")
            self._zeros_slot(v, "h", "h")

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon, var.dtype.base_dtype)

        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)

        beta1_fix = 1 - beta1_power
        beta2_fix = 1 - beta2_power

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        h = self.get_slot(var, "h")

        m_t = state_ops.assign(m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking)
        v_t = state_ops.assign(v, beta2_t * v + (grad * grad) * (1 - beta2_t), use_locking=self._use_locking)
        h_t = state_ops.assign(h, tf.maximum(h, v_t / beta2_fix), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix) / (math_ops.sqrt(h_t) + epsilon_t), use_locking=self._use_locking)
        return var_update

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1, use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(self._beta2_power * self._beta2, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2], name=name_scope)


class AdaShift(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, keep_num=10, func=lambda x: tf.reduce_mean(x), use_locking=False, name="AdaShift"):

        super(AdaShift, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._first_grad_weight = beta1 ** (keep_num - 1) / sum(beta1 ** i for i in range(keep_num))
        self._last_grad_weight = 1. / sum(beta1 ** i for i in range(keep_num))

        self._func = func
        self._step = None
        self._keep_num = 1 if beta1 == 0.0 else keep_num

    def _create_slots(self, var_list):

        self._step = tf.Variable(0, name="step", trainable=False, dtype=tf.int64)

        for v in var_list:
            self._zeros_slot(v, "m", "m")
            self._zeros_slot(v, "v", "v")
            self._get_or_make_slot_with_initializer(v, tf.zeros_initializer(), tf.TensorShape([self._keep_num] + v.get_shape().as_list()), v.dtype, "g", "g")

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon, var.dtype.base_dtype)

        first_grad_weight_t = math_ops.cast(self._first_grad_weight, var.dtype.base_dtype)
        last_grad_weight_t = math_ops.cast(self._last_grad_weight, var.dtype.base_dtype)

        step = self._step
        keep_num = self._keep_num

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        g = self.get_slot(var, "g")

        idx = tf.mod(step, keep_num)
        g0 = tf.gather(g, idx)

        v_t = state_ops.assign(v, v * beta2_t + self._func(tf.square(g0)) * (1 - beta2_t), use_locking=self._use_locking)
        m_t = state_ops.assign(m, (m - first_grad_weight_t * g0) * beta1_t + last_grad_weight_t * grad, use_locking=self._use_locking) if self._beta1 > 0 else grad

        with tf.control_dependencies([v_t, m_t]):
            g_t = tf.scatter_update(g, idx, grad, use_locking=self._use_locking)

        beta2_fix = 1.0 - tf.pow(beta2_t, tf.maximum(1.0, tf.cast(step, tf.float32) - (keep_num - 1.0)))
        delta = lr_t * m_t / (math_ops.sqrt(v_t / beta2_fix) + epsilon_t) * tf.cast(step >= keep_num, tf.float32)
        var_update = state_ops.assign_sub(var, delta, use_locking=self._use_locking)

        # if '/x32/r0/conv1/conv2d/w:0' in var.name:
        #     # g_t = tf.Print(g_t, [tf.reshape(grad, [-1])[0]], "grad")
        #     # g_t = tf.Print(g_t, [tf.reshape(m_t, [-1])[0]], "m_t")
        #     # g_t = tf.Print(g_t, [tf.reshape(g0, [-1])[0]], "g0")
        #     # g_t = tf.Print(g_t, [tf.reshape(math_ops.sqrt(v_t / beta2_fix) + epsilon_t, [-1])[0]], "v_t")
        #     g_t = tf.Print(g_t, [tf.norm(delta)], "delta")
        # g_t = tf.Print(g_t, [tf.norm(var)], var.name)

        return control_flow_ops.group(*[var_update, g_t])

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._step):
                update_step = self._step.assign(self._step + 1, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_step], name=name_scope)


def tf_safe_norm(x, axis=None, keepdims=False, epsilon=1e-10):
    return tf.sqrt(epsilon + tf.reduce_sum(tf.square(x), axis=axis, keepdims=keepdims))


class AdaShiftNW(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, keep_num=10, scale_var=False, use_locking=False, name="AdaShiftNW"):

        super(AdaShiftNW, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._first_grad_weight = beta1 ** (keep_num - 1) / sum(beta1 ** i for i in range(keep_num))
        self._last_grad_weight = 1. / sum(beta1 ** i for i in range(keep_num))

        self._step = None
        self._keep_num = 1 if beta1 == 0.0 else keep_num
        self._scale_var = scale_var

    def _create_slots(self, var_list):

        self._step = tf.Variable(0, name="step", trainable=False, dtype=tf.int64)

        for v in var_list:
            self._zeros_slot(v, "m", "m")
            self._zeros_slot(v, "v", "v")
            self._get_or_make_slot_with_initializer(v, tf.zeros_initializer(), tf.TensorShape([self._keep_num] + v.get_shape().as_list()), v.dtype, "g", "g")

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon, var.dtype.base_dtype)

        first_grad_weight_t = math_ops.cast(self._first_grad_weight, var.dtype.base_dtype)
        last_grad_weight_t = math_ops.cast(self._last_grad_weight, var.dtype.base_dtype)

        step = self._step
        keep_num = self._keep_num

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        g = self.get_slot(var, "g")

        idx = tf.mod(step, keep_num)
        g0 = tf.gather(g, idx)

        v_t = state_ops.assign(v, tf.maximum(v * beta2_t, tf.norm(g0)), use_locking=self._use_locking)
        m_t = state_ops.assign(m, (m - first_grad_weight_t * g0) * beta1_t + last_grad_weight_t * grad, use_locking=self._use_locking) if self._beta1 > 0 else grad

        with tf.control_dependencies([v_t, m_t]):
            g_t = tf.scatter_update(g, idx, grad, use_locking=self._use_locking)

        delta = lr_t * m_t / (v_t + epsilon_t) * tf.cast(step >= keep_num, tf.float32)

        if self._scale_var:
            delta *= tf.norm(var)

        var_update = state_ops.assign_sub(var, delta, use_locking=self._use_locking)

        # if '/x32/r0/conv1/conv2d/w:0' in var.name:
        #     # g_t = tf.Print(g_t, [tf.reshape(grad, [-1])[0]], "grad")
        #     # g_t = tf.Print(g_t, [tf.reshape(m_t, [-1])[0]], "m_t")
        #     # g_t = tf.Print(g_t, [tf.reshape(g0, [-1])[0]], "g0")
        #     # g_t = tf.Print(g_t, [tf.reshape(v_t + epsilon_t, [-1])[0]], "v_t")
        #     g_t = tf.Print(g_t, [tf.norm(var)], "var")
        # g_t = tf.Print(g_t, [tf.norm(var)], var.name)

        return control_flow_ops.group(*[var_update, g_t])

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._step):
                update_step = self._step.assign(self._step + 1, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_step], name=name_scope)