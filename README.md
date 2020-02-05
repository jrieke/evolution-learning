# Biologically Plausible Deep Learning through Neuroevolution

This is the code for my master thesis at TU Berlin and BCCN Berlin.

It combines an evolutionary approach with different learning algorithms
(backpropagation, feedback alignment, Hebbian learning, last-layer learning)
on MNIST. You can read the complete thesis
[here](master-thesis-johannes-rieke-final.pdf).

![](figures/circles.png)


## Abstract

Recently, deep learning has been increasingly used as a framework to understand information processing in the brain. Most existing research in this area has focused on finding biologically plausible learning algorithms. In this thesis, we investigate how the architecture of a neural network interacts with such learning algorithms. Inspired by the intricate wiring of biological brains, we use an evolutionary algorithm to develop network architectures by mutating neurons and connections. Within the evolution loop, the synaptic weights of the networks are trained with one of four learning algorithms, which vary in complexity and biological plausibility (backpropagation, feedback alignment, Hebbian learning, last-layer learning). We investigate how the evolved network architectures differ between learning algorithms, both in terms of their performance on image classification with MNIST as well as their topologies. We show that for all learning algorithms, the evolved architectures learn much better than random networks – i.e. evolution and learning interact. Also, we find that more complex learning algorithms (e.g. backpropagation) can make use of larger, deeper, and more densely connected networks, which achieve higher performances. For simpler and biologically more plausible learning algorithms (e.g. Hebbian learning), however, the evolution process prefers smaller networks. Our results demonstrate that the architecture of a neural network plays an important role in its ability to learn. Using a range of learning algorithms, we show that evolutionary processes can find architectures of artificial neural networks that are particularly suited for learning – similar to how biological evolution potentially shaped the complex wiring of neurons in the brain.


## Citation

If you use this code, please cite my thesis:

    @mastersthesis{Rieke2020,
        author = {Johannes Rieke},
        title = {Biologically Plausible Deep Learning through Neuroevolution},
        school = {Technical University of Berlin},
        year = 2020,
        url = {https://github.com/jrieke/evolution-learning}
    }
