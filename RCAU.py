import tensorflow as tf
import tensorflow.contrib.slim as slim

class RCAU_net( object ):

    def __init__(self):
        self.startflag = True

    def RCAU_net_begin(self,images):

        with tf.variable_scope('RCAU_path'):

            new_state = tf.zeros( [14,14,512],dtype=tf.float32 )
            batch_size = images.get_shape()[0].value

            for index in range( batch_size ):

                if index == 0:
                    new_state = images[index,:,:,:]
                    new_state = tf.expand_dims(new_state,0)
                    continue

                image = images[index,:,:,:]
                image = tf.expand_dims(image, 0)
                new_state = self.forward( image = image, s_state = new_state)

                if self.startflag == True:
                    tf.get_variable_scope().reuse_variables()
                    self.startflag = False

            return new_state

    def forward( self, image, s_state ):

        res = s_state - image

        #####################
        #  attention        #
        #####################
        s_res = self.s_attention_g( x = res ) #Spatial Attention
        c_res = self.c_attention_g( x = res ) #Channel Attention
        attention_res = s_res * c_res
        attention_res = tf.nn.sigmoid( attention_res )

        #####################
        #  reset gate D     #
        #####################
        f_res = tf.concat( (image,s_state), axis = 3 )
        f_res = self.conv_block( f_res,512,scope="layer_4" )
        f_res = tf.nn.sigmoid( f_res )
        f_res = f_res * image

        #####################
        #  add              #
        #####################
        res1 = attention_res * s_state
        res2 = ( 1.0 - attention_res ) * f_res
        final_res = res1 + res2

        return final_res

    def s_attention_g(self,x):

        x = tf.reduce_mean( x, axis=3 )
        x = tf.expand_dims( x, axis=3 )
        x = self.conv_block( inputs = x, n_filters = 256,scope="layer_1" )
        x = tf.nn.relu(x)
        x = self.conv_block( inputs = x, n_filters = 1,scope="layer_2" )
        return x

    def c_attention_g(self,x):

        x = slim.avg_pool2d( x,2 )
        x = self.conv_block( x,512,kernel_size = [3,3],stride=7,scope="layer_3")
        return x

    def conv_block(self, inputs, n_filters, kernel_size=[3, 3],stride = 1,scope=None ):

        conv = slim.conv2d(inputs, n_filters, kernel_size,stride=stride, activation_fn=None, normalizer_fn=None,scope=scope)
        return conv

