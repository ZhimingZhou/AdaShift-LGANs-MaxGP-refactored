from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import numpy as np


class MyOptimizer(optimizer.Optimizer):

    def _apply_sparse(self, grad, var):
        grad_t = self._zeros_slot(var, "g", self._name)
        grad_t = state_ops.assign(grad_t, tf.zeros_like(grad_t), use_locking=self._use_locking) # assign_sub ? bug?
        grad_t = state_ops.scatter_add(grad_t, grad.indices, grad.values, use_locking=self._use_locking)
        return self._apply_dense(grad_t, var)


class Grad(MyOptimizer):

    def __init__(self, learning_rate=0.001, use_locking=False, name="myGrad"):
        super(Grad, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._lr_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        var_update = state_ops.assign_sub(var, lr_t * grad)
        return var_update


class Mom(MyOptimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, use_locking=False, name="myMom"):
        super(Mom, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1

        self._lr_t = None
        self._beta1_t = None

        self._beta1_power = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")

    def _create_slots(self, var_list):

        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and tf.contrib.eager.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", "m")

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta1_fix = 1.0 - beta1_power

        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix), use_locking=self._use_locking)
        return var_update

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1], name=name_scope)


class AdaMax(MyOptimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, use_locking=False, name="AdaMax"):
        super(AdaMax, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        self._beta1_power = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):

        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and tf.contrib.eager.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", "m")
            self._zeros_slot(v, "v", "v")

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta1_fix = 1 - beta1_power

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_t = state_ops.assign(m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking)
        v_t = state_ops.assign(v, tf.maximum(beta2_t * v, grad * grad), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix) / (math_ops.sqrt(v_t) + epsilon_t), use_locking=self._use_locking)
        return var_update

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1], name=name_scope)


class Adam(MyOptimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, use_locking=False, name="myAdam"):
        super(Adam, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        self._beta1_power = None
        self._beta2_power = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):

        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and tf.contrib.eager.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)
                self._beta2_power = tf.Variable(self._beta2, name="beta2_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", "m")
            self._zeros_slot(v, "v", "v")

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

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
                update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(self._beta2_power * self._beta2_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2], name=name_scope)


class AMSGrad(MyOptimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, use_locking=False, name="myAMSGrad"):
        super(AMSGrad, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        self._beta1_power = None
        self._beta2_power = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):

        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and tf.contrib.eager.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)
                self._beta2_power = tf.Variable(self._beta2, name="beta2_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", "m")
            self._zeros_slot(v, "v", "v")
            self._zeros_slot(v, "h", "h")

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

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
                update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(self._beta2_power * self._beta2_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2], name=name_scope)


class AdaShift(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, op='max_mean', keep_num=10, mov_num=None, use_locking=False, name="AdaShift"):

        super(AdaShift, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._op = op
        self._keep_num = 1 if beta1 == 0.0 else keep_num
        self._mov_num = self._keep_num if mov_num is None else min(self._keep_num, mov_num)

        s = [tf.pow(self._beta1, self._mov_num-i-1) for i in range(self._mov_num)]
        sum = tf.reduce_sum(s)
        self.s = [s[i] / sum for i in range(self._mov_num)]
        
        self._lr_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):

        self.first_var = min(var_list, key=lambda x: x.name)

        for v in var_list:
            for i in range(self._keep_num+1):
                self._zeros_slot(v, "g%d" % i, "g%d" % i)
            self._zeros_slot(v, "v", "v")
            self._zeros_slot(v, "z", "z")
            self._zeros_slot(v, "b2p", "b2p")

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        g = [self.get_slot(var, "g%d" % i) for i in range(self._keep_num+1)]
        v = self.get_slot(var, "v")
        z = self.get_slot(var, "z")
        b2p = self.get_slot(var, "b2p")

        if self._op == 'none':
            v_t = state_ops.assign(v, v * beta2_t + tf.square(g[0]) * (1 - beta2_t), use_locking=self._use_locking)
        elif self._op == 'mean':
            v_t = state_ops.assign(v, v * beta2_t + tf.reduce_mean(tf.square(g[0])) * (1 - beta2_t), use_locking=self._use_locking)
        elif self._op == 'max':
            v_t = state_ops.assign(v, v * beta2_t + tf.reduce_max(tf.square(g[0])) * (1 - beta2_t), use_locking=self._use_locking)
        elif self._op == 'max_none':
            v_t = state_ops.assign(v, tf.maximum(v * beta2_t, tf.square(g[0])), use_locking=self._use_locking)
        elif self._op == 'max_mean':
            v_t = state_ops.assign(v, tf.maximum(v * beta2_t, tf.reduce_mean(tf.square(g[0]))), use_locking=self._use_locking)
        elif self._op == 'max_max':
            v_t = state_ops.assign(v, tf.maximum(v * beta2_t, tf.reduce_max(tf.square(g[0]))), use_locking=self._use_locking)
        else:
            assert False

        with ops.control_dependencies([v_t]):
            g_t = state_ops.assign(g[-1], grad, use_locking=self._use_locking)
            for i in range(self._keep_num):
                with ops.control_dependencies([g_t]):
                    g_t = state_ops.assign(g[i], g[i + 1], use_locking=self._use_locking)

        with ops.control_dependencies([g_t]):
            m_t = tf.reduce_sum([g[-i-1]*self.s[-i-1] for i in range(self._mov_num)], axis=0)

        with ops.control_dependencies([v_t]):
            z_t = state_ops.assign(z, tf.cast(tf.logical_or(v_t > 0.0, z > 0.0), tf.float32))

        if self._op == 'max_max' or self._op == 'max_mean' or self._op == 'max_none':
            step_t = z_t * m_t / (math_ops.sqrt(v_t) + epsilon_t)
        else:
            b2p_t = state_ops.assign(b2p, b2p * beta2_t * tf.sign(z_t) + (1.0 - tf.sign(z_t)), use_locking=self._use_locking)
            b2_fix = tf.maximum(1e-8, 1.0 - b2p_t)
            step_t = z_t * m_t / (math_ops.sqrt(v_t / b2_fix) + epsilon_t)

        var_update = state_ops.assign_sub(var, lr_t * step_t, use_locking=self._use_locking)
        return var_update

    def _finish(self, update_ops, name_scope):
        return control_flow_ops.group(*update_ops, name=name_scope)


class AdamW(MyOptimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, use_locking=False, name="AdamW"):
        super(AdamW, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        self._beta1_power = None
        self._beta2_power = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):

        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and tf.contrib.eager.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)
                self._beta2_power = tf.Variable(self._beta2, name="beta2_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", "m")
            self._zeros_slot(v, "v", "v")

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)

        beta1_fix = 1 - beta1_power
        beta2_fix = 1 - beta2_power

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_t = state_ops.assign(m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking)
        v_t = state_ops.assign(v, beta2_t * v + (grad * grad) * (1 - beta2_t), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * tf.maximum(tf.abs(var), 0.01) * (m_t / beta1_fix) / (math_ops.sqrt(v_t / beta2_fix) + epsilon_t), use_locking=self._use_locking)
        return var_update

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(self._beta2_power * self._beta2_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2], name=name_scope)
