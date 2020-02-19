import torch

from layers import LinearLayer, LinearLayerFA


def test_LinearLayer():
    """
    Test LinearLayer by comparing the network's computation to basic matrix multiplication.

    Here we generate a set of random inputs and weights, and compare
    the output of the feedforward path as well as the computed
    gradient for the weights in our own implementation and pythorch's
    native one.
    """

    # Generate input data.
    images = torch.randn(2, 10)

    # Generate initial weights and masks.
    weights_mask = torch.randn(10, 10)
    weights_mask[weights_mask < 0] = 0
    weights_mask[weights_mask > 0] = 1
    initial_weights = torch.randn(10, 10) * weights_mask
    assert initial_weights.grad is None

    ############## our method ################
    weights_ours = initial_weights.clone()
    weights_ours.requires_grad = True

    # Create the layer and compute the output.
    layer = LinearLayer(weights_ours, weights_mask)
    output_ours = layer.forward(images)
    loss_ours = ((output_ours - 1) ** 2).mean()
    loss_ours.backward()
    grad_ours = weights_ours.grad

    ############## pytorch's ################
    weights_py = initial_weights.clone()
    weights_py.requires_grad = True

    # Create output without using our own layer implementation.
    def make_backward_hook(weight_mask):
        """ Helper function to create a backward hook for masking 
        gradients.
        """
        return lambda grad: grad * weight_mask

    weights_py.register_hook(make_backward_hook(weights_mask))
    output_py = torch.mm(images, weights_py.t())
    loss_py = ((output_py - 1) ** 2).mean()
    loss_py.backward()
    grad_py = weights_py.grad

    ############# compare ################
    assert torch.all(torch.eq(output_ours, output_py))
    assert torch.all(torch.eq(loss_ours, loss_py))
    assert torch.all(torch.eq(grad_ours, grad_py))


def test_LinearLayerFA():
    """
    Test the feedforward method of LinearLayerFA by making sure
    that the backward weights stay fixed after optimizing.
    """

    num_inputs = 10

    # Generate weights and masks.
    weights_mask = torch.randn(10, 10)
    weights_mask[weights_mask < 0] = 0
    weights_mask[weights_mask > 0] = 1

    weights_ff = torch.randn(10, 10) * weights_mask
    weights_fb = torch.randn(10, 10) * weights_mask
    weights_ff.requires_grad = True

    weights_ff_initial = weights_ff.clone().detach()
    weights_fb_initial = weights_fb.clone().detach()

    # Create the layer and compute the output.
    layer = LinearLayerFA(weights_ff, weights_fb,weights_mask)
    opt = torch.optim.SGD([layer.weight_matrix], lr=0.01)

    for i in range(num_inputs):
        opt.zero_grad()

        # Generate input data
        images = torch.randn(2, 10)

        output = layer.forward(images)
        loss = ((output - 1) ** 2).mean()
        loss.backward()
        opt.step()

    assert torch.all(torch.eq(weights_fb_initial, weights_fb.data))

def test_LinearFunctionFA():
    """Make sure that LinearLayerFA with symmetric weights behaves in the same
    way as LinearLayer."""

    num_inputs = 10

    # Generate weights and masks.
    weights_mask = torch.randn(10, 10)
    weights_mask[weights_mask < 0] = 0
    weights_mask[weights_mask > 0] = 1

    weights_ff = torch.randn(10, 10) * weights_mask
    weights_fb = weights_ff.clone()
    weights_ff.requires_grad = True
    weights_ff_initial = weights_ff.clone().detach()

    # Create the layer and compute the output.
    layerFA = LinearLayerFA(weights_ff, weights_fb, weights_mask)
    optFA = torch.optim.SGD([layerFA.weight_matrix], lr=0.01)

    layer = LinearLayer(weights_ff, weights_mask)
    opt = torch.optim.SGD([layer.weight_matrix], lr=0.01)

    for i in range(num_inputs):
        opt.zero_grad()
        optFA.zero_grad()

        # Generate input data
        images = torch.randn(2, 10)

        output = layer.forward(images)
        outputFA = layerFA.forward(images)

        loss = ((output - 1) ** 2).mean()
        lossFA = ((outputFA - 1) ** 2).mean()

        loss.backward()
        lossFA.backward()

        opt.step()
        optFA.step()

        assert torch.all(torch.eq(output, outputFA))
        assert loss == lossFA
        assert torch.all(torch.eq(layerFA.weight_matrix, layer.weight_matrix))
        assert torch.all(torch.eq(layerFA.weight_matrix.grad, \
                                            layer.weight_matrix.grad))
