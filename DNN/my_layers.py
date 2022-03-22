import keras.backend as K
import tensorflow as tf
from tensorflow.keras import initializers, regularizers

from tensorflow.keras import layers
# from keras.layers.convolutional import Convolution1D
import tensorflow.keras as keras
from tensorflow.keras.layers import Flatten, GlobalMaxPooling1D, Dense, Convolution1D, Dropout,\
							 GlobalAveragePooling1D, Concatenate, Layer, Add
# from keras.engine.topology import Layer
import numpy as np
import sys


class BaseLayer(keras.layers.Layer):
    def build_layers(self, input_shape):
        shape = input_shape
        for layer in self.layers:
            layer.build(shape)
            shape=layer.compute_output_shape(shape)


class ExpertModule_trm(keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.conv_layers = []
        self.pooling_layers = []
        self.shapes=[]
        # self.filter_size=[5,3]
        self.filters=128
        self.layers = []
        super(ExpertModule_trm, self).__init__(**kwargs)


    def build(self, input_shape):
        self.layers.append(MultiHeadAttention(4, 100))
        self.layers.append(Dropout(0.1))
        self.layers.append(Dense(400, activation='relu'))
        self.layers.append(GlobalMaxPooling1D())
        self.layers.append(GlobalAveragePooling1D())
        self.layers.append(Concatenate())
        self.layers.append(Dropout(0.1))
        self.layers.append(Dense(self.units[0], activation='relu'))
        self.layers.append(Dense(self.units[1], activation='relu'))
        self.layers.append(Dropout(0.1))
        # self.layers.append(Add())


        super(ExpertModule_trm,self).build(input_shape)


    def call(self, inputs):
        xs = self.layers[0](inputs)
        xs = self.layers[1](xs)
        xs = self.layers[2](xs)
        xs_max = self.layers[3](xs)
        xs_avg = self.layers[4](xs)
        xs = self.layers[5]([xs_max, xs_avg])
        for layer in self.layers[6:]:
            xs=layer(xs)
        return xs

    def compute_output_shape(self, input_shape):
        return input_shape[0]+[self.units[1]]



class GateModule(BaseLayer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.conv_layers = []
        self.pooling_layers=[]
        self.layers = []
        super(GateModule, self).__init__(**kwargs)

    def build(self, input_shape):
        self.layers.append(MultiHeadAttention(4, 100))
        self.layers.append(Dropout(0.1))
        self.layers.append(Dense(400, activation='relu'))
        self.layers.append(GlobalMaxPooling1D())
        self.layers.append(GlobalAveragePooling1D())
        self.layers.append(Concatenate())
        self.layers.append(Dropout(0.1))
        self.layers.append(Dense(self.units[0], activation='relu'))
        self.layers.append(Dense(self.units[0], activation='relu'))
        self.layers.append(Dropout(0.1))
        self.layers.append(Dense(self.units[1], activation='softmax'))


        super(GateModule,self).build(input_shape)


    def call(self, inputs):
        xs = self.layers[0](inputs)
        xs = self.layers[1](xs)
        xs = self.layers[2](xs)
        xs_max = self.layers[3](xs)
        xs_avg = self.layers[4](xs)
        xs = self.layers[5]([xs_max, xs_avg])
        for layer in self.layers[6:]:
            xs=layer(xs)
        return xs

    def compute_output_shape(self, input_shape):
        return input_shape[0]+[self.units[-1]]

class HSMMBottom(BaseLayer):
    # Hate Speech Mixture Model
    def __init__(self,
                 model_type,
                 non_gate,
                 expert_units,
                 gate_unit=100,
                 task_num=2, expert_num=3,
                 **kwargs):
        self.model_type = model_type
        self.non_gate = non_gate
        self.gate_unit = gate_unit
        self.expert_units = expert_units
        self.task_num = task_num
        self.expert_num = expert_num
        self.experts=[]
        self.gates=[]
        super(HSMMBottom, self).__init__(**kwargs)

    def build(self,input_shape):
        if self.model_type in {'HHMM_word_char', 'HHMM_word_char1'}:
            for i in range(self.expert_num):
                expert = ExpertModule_word_char(units=self.expert_units)
                expert.build(input_shape)
                self.experts.append(expert)
            for i in range(self.task_num):
                gate = GateModule_word_char(units=[self.gate_unit, self.expert_num])
                gate.build(input_shape)
                self.gates.append(gate)
        elif self.model_type == 'HHMM_transformer':
            for i in range(self.expert_num):
                expert = ExpertModule_trm(units=self.expert_units)
                expert.build(input_shape)
                self.experts.append(expert)
            for i in range(self.task_num):
                gate = GateModule(units=[self.gate_unit, self.expert_num])
                gate.build(input_shape)
                self.gates.append(gate)
        else:
            for i in range(self.expert_num):
                expert = ExpertModule(units=self.expert_units)
                expert.build(input_shape)
                self.experts.append(expert)
            for i in range(self.task_num):
                gate = GateModule_char(units=[self.gate_unit, self.expert_num])
                gate.build(input_shape)
                self.gates.append(gate)
        super(HSMMBottom,self).build(input_shape)

    def call(self, inputs):
        # 构建多个expert
        expert_outputs=[]
        for expert in self.experts:
            expert_outputs.append(expert(inputs))

        # 构建多个gate，用来加权expert
        gate_outputs=[]
        if self.non_gate:
            print('1111111111111111111111111无门控')
            self.expert_output = tf.stack(expert_outputs,axis=1) # batch_size, expert_num, expert_out_dim
            m1 = tf.reduce_mean(self.expert_output, axis=1)
            outputs = tf.stack([m1, m1], axis=1)
            return outputs
            
        else:
            for gate in self.gates:
                gate_outputs.append(gate(inputs))
            # 使用gate对expert进行加权平均
            self.expert_output=tf.stack(expert_outputs,axis=1) # batch_size, expert_num, expert_out_dim
            self.gate_output=tf.stack(gate_outputs,axis=1) # batch_size, task_num, expert_num
            outputs=tf.matmul(self.gate_output,self.expert_output) # batch_size,task_num,expert_out_dim
            return outputs

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.task_num, self.expert_units[-1]]

class HSMMTower(BaseLayer):
    # Hate Speech Mixture Model Tower
    def __init__(self,
                 units,
                 **kwargs):
        self.units = units
        self.layers=[]
        super(HSMMTower, self).__init__(**kwargs)

    def build(self, input_shape):
        for unit in self.units[:-1]:
            self.layers.append(Dense(unit, activation='relu'))
        self.layers.append(Dropout(0.1))
        self.layers.append(Dense(self.units[-1], activation='softmax'))
        self.build_layers(input_shape)
        super(HSMMTower,self).build(input_shape)

    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.units[-1]]



class MultiHeadAttention(Layer):
	"""多头注意力机制
	"""
	def __init__(self,heads, head_size, output_dim=None, **kwargs):
		self.heads = heads
		self.head_size = head_size
		self.output_dim = output_dim or heads * head_size
		super(MultiHeadAttention, self).__init__(**kwargs)

	def build(self, input_shape):
		# 为该层创建一个可训练的权重
		#inputs.shape = (batch_size, time_steps, seq_len)
		self.kernel = self.add_weight(name='kernel',
									  shape=(3,input_shape[2], self.head_size),
									  initializer='uniform',
									  trainable=True)
		self.dense = self.add_weight(name='dense',
									  shape=(input_shape[2], self.output_dim),
									  initializer='uniform',
									  trainable=True)

		super(MultiHeadAttention, self).build(input_shape)  # 一定要在最后调用它

	def call(self, x):
		out = []
		for i in range(self.heads):
			WQ = K.dot(x, self.kernel[0])
			WK = K.dot(x, self.kernel[1])
			WV = K.dot(x, self.kernel[2])

			# print("WQ.shape",WQ.shape)
			# print("K.permute_dimensions(WK, [0, 2, 1]).shape",K.permute_dimensions(WK, [0, 2, 1]).shape)

			QK = K.batch_dot(WQ,K.permute_dimensions(WK, [0, 2, 1]))
			QK = QK / (100**0.5)
			QK = K.softmax(QK)

			# print("QK.shape",QK.shape)

			V = K.batch_dot(QK,WV)
			out.append(V)
		out = Concatenate(axis=-1)(out)
		out = K.dot(out, self.dense)
		return out

	def compute_output_shape(self, input_shape):
		return (input_shape[0],input_shape[1],self.output_dim)

#     """多头注意力机制
#     """
#     def __init__(
#         self,
#         heads,
#         head_size,
#         out_dim=None,
#         key_size=None,
#         use_bias=True,
#         attention_scale=True,
#         kernel_initializer='glorot_uniform',
#         **kwargs
#     ):
#         super(MultiHeadAttention_from_bert4keras, self).__init__(**kwargs)
#         self.heads = heads
#         self.head_size = head_size
#         self.out_dim = out_dim or heads * head_size
#         self.key_size = key_size or head_size
#         self.use_bias = use_bias
#         self.attention_scale = attention_scale
#         self.kernel_initializer = initializers.get(kernel_initializer)

#     def build(self, input_shape):
#         super(MultiHeadAttention_from_bert4keras, self).build(input_shape)
#         self.q_dense = Dense(
#             units=self.key_size * self.heads,
#             use_bias=self.use_bias,
#             kernel_initializer=self.kernel_initializer
#         )
#         self.k_dense = Dense(
#             units=self.key_size * self.heads,
#             use_bias=self.use_bias,
#             kernel_initializer=self.kernel_initializer
#         )
#         self.v_dense = Dense(
#             units=self.head_size * self.heads,
#             use_bias=self.use_bias,
#             kernel_initializer=self.kernel_initializer
#         )
#         self.o_dense = Dense(
#             units=self.out_dim,
#             use_bias=self.use_bias,
#             kernel_initializer=self.kernel_initializer
#         )

#     # @recompute_grad
#     def call(self, inputs, mask=None, a_mask=None, p_bias=None):
#         """实现多头注意力
#         q_mask: 对输入的query序列的mask。
#                 主要是将输出结果的padding部分置0。
#         v_mask: 对输入的value序列的mask。
#                 主要是防止attention读取到padding信息。
#         a_mask: 对attention矩阵的mask。
#                 不同的attention mask对应不同的应用。
#         p_bias: 在attention里的位置偏置。
#                 一般用来指定相对位置编码的种类。
#         """
#         q, k, v = inputs[:3]
#         q_mask, v_mask, n = None, None, 3
#         if mask is not None:
#             if mask[0] is not None:
#                 q_mask = K.cast(mask[0], K.floatx())
#             if mask[2] is not None:
#                 v_mask = K.cast(mask[2], K.floatx())
#         if a_mask:
#             a_mask = inputs[n]
#             n += 1
#         # 线性变换
#         qw = self.q_dense(q)
#         kw = self.k_dense(k)
#         vw = self.v_dense(v)
#         # 形状变换
#         qw = K.reshape(qw, (-1, K.shape(q)[1], self.heads, self.key_size))
#         kw = K.reshape(kw, (-1, K.shape(k)[1], self.heads, self.key_size))
#         vw = K.reshape(vw, (-1, K.shape(v)[1], self.heads, self.head_size))
#         # Attention
#         a = tf.einsum('bjhd,bkhd->bhjk', qw, kw)
#         # 处理位置编码
#         if p_bias == 'typical_relative':
#             pos_embeddings = inputs[n]
#             a = a + tf.einsum('bjhd,jkd->bhjk', qw, pos_embeddings)
#         elif p_bias == 't5_relative':
#             pos_embeddings = K.permute_dimensions(inputs[n], (2, 0, 1))
#             a = a + K.expand_dims(pos_embeddings, 0)
#         # Attention（续）
#         if self.attention_scale:
#             a = a / self.key_size**0.5
#         a = sequence_masking(a, v_mask, 1, -1)
#         if a_mask is not None:
#             a = a - (1 - a_mask) * 1e12
#         a = K.softmax(a)
#         # 完成输出
#         o = tf.einsum('bhjk,bkhd->bjhd', a, vw)
#         if p_bias == 'typical_relative':
#             o = o + tf.einsum('bhjk,jkd->bjhd', a, pos_embeddings)
#         o = K.reshape(o, (-1, K.shape(o)[1], self.head_size * self.heads))
#         o = self.o_dense(o)
#         # 返回结果
#         o = sequence_masking(o, q_mask, 0)
#         return o

#     def compute_output_shape(self, input_shape):
#         return (input_shape[0][0], input_shape[0][1], self.out_dim)

#     def compute_mask(self, inputs, mask=None):
#         if mask is not None:
#             return mask[0]

#     def get_config(self):
#         config = {
#             'heads': self.heads,
#             'head_size': self.head_size,
#             'out_dim': self.out_dim,
#             'key_size': self.key_size,
#             'use_bias': self.use_bias,
#             'attention_scale': self.attention_scale,
#             'kernel_initializer':
#                 initializers.serialize(self.kernel_initializer),
#         }
#         base_config = super(MultiHeadAttention_from_bert4keras, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))



