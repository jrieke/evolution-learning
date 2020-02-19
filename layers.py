"""
Torch layers and functions for WeightLearningNetwork.

Includes a normal linear layer (adapted for our network structure) and a layer performing feedback
alignment. We cannot use the standard torch layers because due to mutations, our network structure
is very different from a traditional, densely connected network.
"""

from torch.autograd import Function
import torch.nn as nn
import torch


class LinearFunction(Function):
    r"""Implementation of a fully-connected layer w/o activation function.

    Copied from https://git.ee.ethz.ch/henningc/teaching
    This class is a ``Function`` that behaves just like PyTorch's class
    :class:`torch.nn.Linear`. Since this class implements the interface
    :class:`torch.autograd.Function`, we can use it to specify a custom
    backpropagation behavior.

    In this specific case, the ``Function`` shall behave just as in classic
    backpropagation (i.e., it shall behave identical to the proprietory PyTorch
    implementation).

    Assuming column vectors: layer input :math:`\mathbf{a} \in \mathbb{R}^K`, 
    a weight matrix :math:`W \in \mathbb{R}^{N \times K}`, and a masks matrix
    :math:`M \in \mathbb{R}^{N \times K}` this layer simply computes

    .. math::
        :label: eq-single-sample

        \mathbf{z} = W  \mathbf{a} 

    Note, since we want to process mini-batches (containing :math:`B` samples
    each), the input to the :meth:`forward` method is actually a set of samples
    :math:`\mathbf{a}` collected into a matrix
    :math:`A \in \mathbb{R}^{B \times K}`.

    The mathematical operation described for single samples in eq.
    :eq:`eq-single-sample`, is stated for a the case of mini-batches below

    .. math::
        :label: eq-mini-batch

        Z = A W^T

    where :math:`Z \in \mathbb{R}^{B \times N}` is the output matrix.
    """

    @staticmethod
    def forward(ctx, A, W, M):
        r"""Compute the output of a linear layer.

        This method implements eq. :eq:`eq-mini-batch`.

        Args:
            ctx: A context. Should be used to store activations which are needed
                in the backward pass.
            A: A mini-batch of input activations :math:`A`.
            W: The weight matrix :math:`W`.
            M: The weight masks :math:`M`.

        Returns:
            The output activations :math:`Z` as defined by eq.
            :eq:`eq-mini-batch`.
        """
        ctx.save_for_backward(A, W, M)

        Z = torch.mm(A, W.t())

        return Z

    @staticmethod
    def backward(ctx, grad_Z):
        r"""Backpropagate the gradients of :math:`Z` through this linear layer.

        The matrix ``grad_Z``, which we denote by
        :math:`\delta_Z \in \mathbb{R}^{B \times N}`, contains the partial
        derivatives of the scalar loss function with respect to each element
        from the :meth:`forward` output matrix :math:`Z`.

        This method backpropagates the global error (encoded in
        :math:`\delta_Z`) to the input tensors of the :meth:`forward` method,
        essentially computing :math:`\delta_A` and :math:`\delta_W`. Note that
        :math:`\delta_W` is then multiplied by the weight masks :math:`M`,
        essentially performing a backward hook on the gradient of :math:`W`.

        These partial derivatives can be computed as follows:

        .. math::

            \delta_A &= \delta_Z W \\
            \delta_W &= (\delta_Z^T A) \odot M \\

        where :math:`\delta_{Z_{b,:}}` denotes the vector retrieved from the
        :math:`b`-th row of :math:`\delta_Z`.

        Args:
            ctx: See description of argument ``ctx`` of method :meth:`forward`.
            grad_Z: The backpropagated error :math:`\delta_Z`.

        Returns:
            (tuple): Tuple containing:

            - **grad_A**: The derivative of the loss with respect to the input
              activations, i.e., :math:`\delta_A`.
            - **grad_W**: The derivative of the loss with respect to the weight
              matrix, i.e., :math:`\delta_W`.
            - **grad_M**: The derivative of the loss with respect to the masks.
              As the masks are never learned, it is always set to None.

            .. note::
                Gradients for input tensors are only computed if their keyword
                ``requires_grad`` is set to ``True``, otherwise ``None`` is
                returned for the corresponding Tensor.
        """
        A, W, M = ctx.saved_tensors

        grad_A = None
        grad_W = None
        grad_M = None

        # We only need to compute gradients for tensors that are flagged to
        # require gradients!
        if ctx.needs_input_grad[0]:
            grad_A = grad_Z.mm(W)
        if ctx.needs_input_grad[1]:
            grad_W = (grad_Z.t().mm(A)) * M

        return grad_A, grad_W, grad_M


class LinearFunctionFA(Function):
    r"""Implementation of a fully-connected layer w/o activation function which
    implements Feedback Alignment in the feedback path.

    Note that, as opposed to the classical Feedback Alignement formalization,
    we consider B (and not B.t() has the dimensions of W). This is for 
    simplicity when setting symmetric connections and other manipulations.

    See :class:`LinearFunction` for further details.

    Args:
        (...): See :class:`LinearFunction` for further details.
        B: The feedback weights :math:`B`.

    """
    @staticmethod
    def forward(ctx, A, W, B, M):
        r"""Compute the output of a linear layer.

        See :meth:`LinearFunction.forward` for further details.

        """
        ctx.save_for_backward(A, W, B, M)

        Z = torch.mm(A, W.t())

        return Z

    @staticmethod
    def backward(ctx, grad_Z):
        r"""Propagate the gradients of :math:`Z` through this linear layer
        according to the Feedback Alignment method.

        The partial derivatives can be computed as follows:

        .. math::

            \delta_A &= \delta_Z B \\
            \delta_W &= (\delta_Z^T A) \odot M \\

        where :math:`\delta_{Z_{b,:}}` denotes the vector retrieved from the
        :math:`b`-th row of :math:`\delta_Z`.

        Args:
            (...): See :meth:`LinearFunction.backward` for further details.

        Returns:
            (tuple): See :meth:`LinearFunction.backward` for further details.
        """
        A, W, B, M = ctx.saved_tensors

        grad_A = None
        grad_W = None
        grad_B = None
        grad_M = None

        # We only need to compute gradients for tensors that are flagged to
        # require gradients!
        if ctx.needs_input_grad[0]:
            grad_A = grad_Z.mm(B)
        if ctx.needs_input_grad[1]:
            grad_W = (grad_Z.t().mm(A)) * M

        return grad_A, grad_W, grad_B, grad_M


class LinearLayer(nn.Module):
    """Wrapper for ``Function`` :class:`learning.LinearFunction`.

    The interface is inspired by the implementation of class
    :class:`torch.nn.Linear`. 
    Note that it is assumed here that matrix representations are compact.

    Attributes:
        weight_matrix (torch.Tensor): The weight matrix :math:`W` of the layer.
        weight_mask (torch.Tensor): The weight masks :math:`M` of the layer. These
            are zero if the weights don't exist in the network and should not
            be updated in the backward pass.
        from_neurons (list): The indices of neurons having connections to this layers.
        to_neurons (list): The indices of neurons in this layer.
    """

    def __init__(self, weight_matrix, weight_mask, from_neurons=None, to_neurons=None):
        """
        Initialize LinearLayer from existing weight matrix.

        Args:
            Same as attributes above.
        """
        nn.Module.__init__(self)

        self.weight_matrix = weight_matrix
        self.weight_mask = weight_mask

        if from_neurons is None:
            from_neurons = list(range(weight_matrix.shape[1]))
        self.from_neurons = from_neurons

        if to_neurons is None:
            to_neurons = list(range(weight_matrix.shape[0]))
        self.to_neurons = to_neurons

    def forward(self, input):
        """Compute the output activation of a linear layer.

        This method simply applies the
        :class:`learning.LinearFunction` ``Function`` using the internally 
        maintained weights.

        Args:
            input: See description of argument ``A`` of method
                :meth:`learning.LinearFunction.forward`.

        Returns:
            See return value of method 
            :meth:`lib.learning.LinearFunction.forward`.
        """

        return LinearFunction.apply(input, self.weight_matrix, self.weight_mask)

    def get_weight(self, from_neuron, to_neuron):
        """Return the weight of the connection between from_neuron and to_neuron."""
        return self.weight_matrix[self.to_neurons.index(to_neuron),
                                  self.from_neurons.index(from_neuron)].item()

    def get_grad(self, from_neuron, to_neuron):
        """Return the gradient of the connection between from_neuron and to_neuron."""
        return self.weight_matrix.grad[self.to_neurons.index(to_neuron),
                                       self.from_neurons.index(from_neuron)].item()


class LinearLayerFA(LinearLayer):
    """Wrapper for ``Function`` :class:`learning.LinearFunctionFA`.

    The interface is inspired by the implementation of class
    :class:`torch.nn.Linear`.

    Attributes:
        weight_matrix (torch.Tensor): The weight matrix :math:`W` of the layer.
        backward_weight_matrix (torch.Tensor): The backward weight matrix :math:`B` of the layer.
        weight_mask (torch.Tensor): The weight masks :math:`M` of the layer. These
            are zero if the weights don't exist in the network and should not
            be updated in the backward pass.
        from_neurons (list): The neurons having (ff or fb) connections to this layer.
        to_neurons (list): The neurons in this layer.
    """

    def __init__(self, weight_matrix, backward_weight_matrix, weight_mask, from_neurons=None,
                 to_neurons=None,):
        """
        Initialize LinearLayerFA from existing weight matrix and backward weight matrix.

        Args:
            See attributes above.
        """
        super().__init__(weight_matrix, weight_mask, from_neurons=from_neurons,
                         to_neurons=to_neurons)

        self.backward_weight_matrix = backward_weight_matrix
        assert self.backward_weight_matrix.shape == weight_matrix.shape 

    def forward(self, input):
        """Compute the output activation of a linear layer.

        This method simply applies the
        :class:`learning.LinearFunctionFA` ``Function`` using the internally 
        maintained feedforward and feedback weights.

        Args:
            input: See description of argument ``A`` of method
                :meth:`learning.LinearFunctionFA.forward`.

        Returns:
            See return value of method 
            :meth:`lib.learning.LinearFunctionFA.forward`.
        """

        return LinearFunctionFA.apply(input, self.weight_matrix, self.backward_weight_matrix,
                                      self.weight_mask)
