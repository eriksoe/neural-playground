# Library containing a Neural Network model.
# The model is Theano-based.

import numpy.random
import numpy as NP
import theano as T
import theano.tensor as TT

def T_variable_with_shape(shape, name=None):
    return TT.TensorType('floatX', [False]*len(shape))(name)

class NeuralNetwork:
    """State:
    in_shape, out_shape: Shapes of the outer layers.
    hidden_dims: Number of hidden layers.
    weights, biases: Theano shared variables containing the current network weights.
    evaluate_fun: Evaluation function.
    gradient_fun: Gradient function.
    """

    #========== SETUP ==================================================
    def __init__(this, in_shape, out_shape, hidden_dims):
        """in_shape and out_shape are the array shapes of the inputs and outputs, respectively.
        hidden_dims is a list of integers stating the sizes of the hidden layers; the length of hidden_dims is the number of such layers.
        """
        this.in_shape = in_shape
        this.out_shape = out_shape
        this.hidden_dims = hidden_dims

        this.initialize()

    def initialize(this):
        "Initialize the network to a valid but random state."
        this.output_biases = NP.random.random(this.out_shape)
        (this.weight_shapes, this.bias_shapes) = this.weight_and_bias_shapes()
        this.weights = [T.shared(numpy.zeros(shape))
                        for shape in this.weight_shapes]
        this.biases = [T.shared(numpy.zeros(shape))
                       for shape in this.bias_shapes]
        this.build_functions()
        this.randomize()

    def randomize(this):
        "Initialize all weights to random values."
        for w in this.weights:
            shape = w.get_value().shape
            w.set_value(0.01 * NP.random.standard_normal(shape))
        for b in this.biases:
            shape = b.get_value().shape
            b.set_value(0.01 * NP.random.standard_normal(shape))

    def weight_and_bias_shapes(this):
        "Helper function - computes weight shapes suitable for tensordot() use."
        weight_shapes = []
        last_shape = this.in_shape
        dls = this.dest_layer_shapes()
        for dest_layer_shape in dls:
            weight_shape = tuple(list(last_shape) + list(dest_layer_shape))
            weight_shapes.append(weight_shape)
            last_shape = dest_layer_shape
        bias_shapes = dls
        return (weight_shapes, bias_shapes)

    def dest_layer_shapes(this):
        "Helper function."
        return [(dim,) for dim in this.hidden_dims] + [this.out_shape]

    #========== EVALUATION etc. ========================================
    def build_functions(this):
        print "NET| Build - evaluation..."
        #---- Evaluation:
        inputs = T_variable_with_shape(this.in_shape, name='inputs')
        outputs = this.T_evaluate(inputs)
        this.evaluate_fun = T.function([inputs], outputs)

        #---- Cost:
        print "NET| Build - cost..."
        expected_outputs = T_variable_with_shape(this.out_shape, name='outputs')
        cost = this.T_cost(outputs, expected_outputs)
        this.cost_fun = T.function([outputs, expected_outputs], cost)

        #---- Gradient:
        print "NET| Build - gradient..."
        #print "weights = %s" % this.weights
        #print "cost = %s" % cost
        #print(T.pp(cost))
        weight_gradient = TT.grad(cost, this.weights)
        #print "weight_gradient = %s" % weight_gradient
        bias_gradient = TT.grad(cost, this.biases)
        #print "bias_gradient = %s" % bias_gradient
        this.gradient_fun = T.function([inputs, expected_outputs],
                                       weight_gradient + bias_gradient)

        #---- Derivatives along given direction:
        print "NET| Build - derivatives..."
        w_direction = [T_variable_with_shape(shape,'w_dir') for shape in this.weight_shapes]
        b_direction = [T_variable_with_shape(shape,'b_dir') for shape in this.bias_shapes]
        step = TT.scalar()
        direction_givens = ([(w, w + step*d) for (w,d) in zip(this.weights, w_direction)] +
                       [(b, b + step*d) for (b,d) in zip(this.biases, b_direction)])
        cost_along_direction = T.clone(cost, replace=dict(direction_givens))
        deriv0 = cost_along_direction
        deriv1 = TT.grad(deriv0, step)
        deriv2 = TT.grad(deriv1, step)
        T.config.exception_verbosity='high'
        this.derivatives_fun = T.function([inputs, expected_outputs] + w_direction + b_direction, [deriv0, deriv1, deriv2], givens=[(step,0.0)])

        #---- Updating:
        print "NET| Build - updating..."
        alpha = TT.scalar('alpha')
        weight_vars = this.weights + this.biases
        delta_vars = ([T_variable_with_shape(shape) for shape in this.weight_shapes] +
                 [T_variable_with_shape(shape) for shape in this.bias_shapes])
        updates = [(v,v+alpha*delta) for (v,delta) in zip(weight_vars, delta_vars)]
        #print "Updates = %s" % updates
        this.update_with_delta_fun = T.function([alpha] + delta_vars, [],
                                                updates=updates)

        print "NET| Build - done."

    def T_evaluate(this, inputs):
        layer_count = len(this.weight_shapes)
        data = inputs
        data_dims = len(this.in_shape)
        for i in range(layer_count):
            if i>0: data = this.logistic(data)

            ws = this.weights[i]
            bs = this.biases[i]
            data = TT.tensordot(data, ws, data_dims) + bs
            data_dims = len(this.bias_shapes[i])
        return data

    def T_cost(this, outputs, expected_outputs):
        return TT.sum((outputs - expected_outputs) ** 2)

    def logistic(this, x):
        return 1 / (1 + TT.exp(-x))

    def gradient_shape(this):
        return this.weight_shapes + this.bias_shapes

    #========== EVALUATION etc. - PUBLIC ==============================
    def evaluate(this, in_data):
        return this.evaluate_fun(in_data)

    def cost(this, in_data, expected_output):
        return this.cost_fun(in_data, expected_output)

    def gradient(this, in_data, expected_output):
        return this.gradient_fun(in_data, expected_output)

    def derivatives(this, in_data, expected_output, direction):
        (d0,d1,d2) = this.derivatives_fun(in_data, expected_output, *direction)
        return (d0,d1,d2)

    def update_with_delta(this, alpha, delta):
        this.update_with_delta_fun(alpha, *delta)



class MiniBatchTrainer:
    def __init__(self, net, minibatch_size=10):
        self._net = net
        self._minibatch_size = minibatch_size

        self._gradient_shape = net.gradient_shape()
        self._reset_minibatch(None)
        self._training_cost = 0.0

    def _reset_minibatch(self, new_gradient):
        self._full_gradient = new_gradient
        self._acc_gradient = [NP.zeros(shape) for shape in self._gradient_shape]
        self._acc_directional_derivatives = (0,0,0)
        self._minibatch_pos = 0

    def present_one_example(self, inputs, outputs):
        #print "Example: %s/%s" % (inputs, outputs)
        net = self._net
        gradient = net.gradient(inputs, outputs)
        self._acc_gradient = [(ag+g) for (ag,g) in zip(self._acc_gradient, gradient)]
        if self._full_gradient != None:
            #print "DB| calcing derivatives from _full_gradient: %s" % (self._full_gradient,)
            (d0,d1,d2) = net.derivatives(inputs, outputs, self._full_gradient)
            (ad0, ad1, ad2) = self._acc_directional_derivatives
            self._acc_directional_derivatives = (ad0+d0, ad1+d1, ad2+d2)
            cur_cost = d0
        else:
            actual_outputs = net.evaluate(inputs)
            cost = net.cost(actual_outputs, outputs)
            cur_cost = cost

        self._training_cost += cur_cost
        print "Cost: %f" % cur_cost

        self._minibatch_pos += 1
        if self._minibatch_pos >= self._minibatch_size:
            self.flush_minibatch()


    def flush_minibatch(self):
        if self._minibatch_pos == 0:
            return

        if self._full_gradient != None:
            print "DB| acc_directional_derivatives = %s" % (self._acc_directional_derivatives,)
            (d0, d1, d2) = self._acc_directional_derivatives
            step = 0.5 * min(abs(d0/d1), abs(d1 / d2))
            step *= (self._minibatch_pos / self._minibatch_size)
            print "Step: %s" % (step,)
            if NP.isfinite(step):
                self._net.update_with_delta(-step, self._full_gradient)
            else:
                print "*** step is %s because of derivativesd= %s" % (step, self._acc_directional_derivatives)

        self._reset_minibatch(self._acc_gradient)
        print "Training cost so far: %s" % (self._training_cost)

