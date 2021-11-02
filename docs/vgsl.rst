.. _vgsl:

VGSL network specification
==========================

kraken implements a dialect of the Variable-size Graph Specification Language
(VGSL), enabling the specification of different network architectures for image
processing purposes using a short definition string.

Basics
------

A VGSL specification consists of an input block, one or more layers, and an
output block. For example:

.. code-block:: console

        [1,48,0,1 Cr3,3,32 Mp2,2 Cr3,3,64 Mp2,2 S1(1x12)1,3 Lbx100 Do O1c103]

The first block defines the input in order of [batch, height, width, channels]
with zero-valued dimensions being variable. Integer valued height or width
input specifications will result in the input images being automatically scaled
in either dimension.

When channels are set to 1 grayscale or B/W inputs are expected, 3 expects RGB
color images. Higher values in combination with a height of 1 result in the
network being fed 1 pixel wide grayscale strips scaled to the size of the
channel dimension.

After the input, a number of layers are defined. Layers operate on the channel
dimension; this is intuitive for convolutional layers but a recurrent layer
doing sequence classification along the width axis on an image of a particular
height requires the height dimension to be moved to the channel dimension,
e.g.:

.. code-block:: console

        [1,48,0,1 S1(1x48)1,3 Lbx100 O1c103]

or using the alternative slightly faster formulation:

.. code-block:: console

        [1,1,0,48 Lbx100 O1c103]

Finally an output definition is appended. When training sequence classification
networks with the provided tools the appropriate output definition is
automatically appended to the network based on the alphabet of the training
data.

Examples
--------

.. code-block:: console

        [1,1,0,48 Lbx100 Do 01c59]

        Creating new model [1,1,0,48 Lbx100 Do] with 59 outputs 
        layer		type	params 
        0		rnn	direction b transposed False summarize False out 100 legacy None
        1		dropout	probability 0.5 dims 1 
        2		linear	augmented False out 59 

A simple recurrent recognition model with a single LSTM layer classifying lines
normalized to 48 pixels in height.

.. code-block:: console

        [1,48,0,1 Cr3,3,32 Do0.1,2 Mp2,2 Cr3,3,64 Do0.1,2 Mp2,2 S1(1x12)1,3 Lbx100 Do 01c59]

        Creating new model [1,48,0,1 Cr3,3,32 Do0.1,2 Mp2,2 Cr3,3,64 Do0.1,2 Mp2,2 S1(1x12)1,3 Lbx100 Do] with 59 outputs 
        layer		type	params 
        0		conv	kernel 3 x 3 filters 32 activation r 
        1		dropout	probability 0.1 dims 2 
        2		maxpool	kernel 2 x 2 stride 2 x 2 
        3		conv	kernel 3 x 3 filters 64 activation r 
        4		dropout	probability 0.1 dims 2 
        5		maxpool	kernel 2 x 2 stride 2 x 2 
        6		reshape from 1 1 x 12 to 1/3 
        7		rnn	direction b transposed False summarize False out 100 legacy None 
        8		dropout	probability 0.5 dims 1 
        9		linear	augmented False out 59 

A model with a small convolutional stack before a recurrent LSTM layer. The
extended dropout layer syntax is used to reduce drop probability on the depth
dimension as the default is too high for convolutional layers. The remainder of
the height dimension (`12`) is reshaped into the depth dimensions before
applying the final recurrent and linear layers.

.. code-block:: console

        [1,0,0,3 Cr3,3,16 Mp3,3 Lfys64 Lbx128 Lbx256 Do 01c59]

        Creating new model [1,0,0,3 Cr3,3,16 Mp3,3 Lfys64 Lbx128 Lbx256 Do] with 59 outputs
        layer		type	params
        0		conv	kernel 3 x 3 filters 16 activation r
        1		maxpool	kernel 3 x 3 stride 3 x 3
        2		rnn	direction f transposed True summarize True out 64 legacy None
        3		rnn	direction b transposed False summarize False out 128 legacy None
        4		rnn	direction b transposed False summarize False out 256 legacy None
        5		dropout	probability 0.5 dims 1
        6		linear	augmented False out 59

A model with arbitrary sized color image input, an initial summarizing
recurrent layer to squash the height to 64, followed by 2 bi-directional
recurrent layers and a linear projection.

Convolutional Layers
--------------------

.. code-block:: console

        C[{name}](s|t|r|l|m)[{name}]<y>,<x>,<d>[,<stride_y>,<stride_x>]
        s = sigmoid
        t = tanh
        r = relu
        l = linear
        m = softmax

Adds a 2D convolution with kernel size `(y, x)` and `d` output channels, applying
the selected nonlinearity. The stride can be adjusted with the optional last
two parameters.

Recurrent Layers
----------------

.. code-block:: console

        L[{name}](f|r|b)(x|y)[s][{name}]<n> LSTM cell with n outputs.
        G[{name}](f|r|b)(x|y)[s][{name}]<n> GRU cell with n outputs.
        f runs the RNN forward only.
        r runs the RNN reversed only.
        b runs the RNN bidirectionally.
        s (optional) summarizes the output in the requested dimension, return the last step.

Adds either an LSTM or GRU recurrent layer to the network using either the `x`
(width) or `y` (height) dimension as the time axis. Input features are the
channel dimension and the non-time-axis dimension (height/width) is treated as
another batch dimension. For example, a `Lfx25` layer on an `1, 16, 906, 32`
input will execute 16 independent forward passes on `906x32` tensors resulting
in an output of shape `1, 16, 906, 25`. If this isn't desired either run a
summarizing layer in the other direction, e.g. `Lfys20` for an input `1, 1,
906, 20`, or prepend a reshape layer `S1(1x16)1,3` combining the height and
channel dimension for an `1, 1, 906, 512` input to the recurrent layer.

Helper and Plumbing Layers
--------------------------

Max Pool
^^^^^^^^
.. code-block:: console

        Mp[{name}]<y>,<x>[,<y_stride>,<x_stride>]

Adds a maximum pooling with `(y, x)` kernel_size and `(y_stride, x_stride)` stride.

Reshape
^^^^^^^

.. code-block:: console

        S[{name}]<d>(<a>x<b>)<e>,<f> Splits one dimension, moves one part to another
                dimension.

The `S` layer reshapes a source dimension `d` to `a,b` and distributes `a` into
dimension `e`, respectively `b` into `f`.  Either `e` or `f` has to be equal to
`d`. So `S1(1, 48)1, 3` on an `1, 48, 1020, 8` input will first reshape into
`1, 1, 48, 1020, 8`, leave the `1` part in the height dimension and distribute
the `48` sized tensor into the channel dimension resulting in a `1, 1, 1024,
48*8=384` sized output. `S` layers are mostly used to remove undesirable non-1
height before a recurrent layer.

.. note::

        This `S` layer is equivalent to the one implemented in the tensorflow
        implementation of VGSL, i.e. behaves differently from tesseract.

Regularization Layers
---------------------

Dropout
^^^^^^^

.. code-block:: console

        Do[{name}][<prob>],[<dim>] Insert a 1D or 2D dropout layer

Adds an 1D or 2D dropout layer with a given probability. Defaults to `0.5` drop
probability and 1D dropout. Set to `dim` to `2` after convolutional layers.

Group Normalization
^^^^^^^^^^^^^^^^^^^

.. code-block:: console

        Gn<groups> Inserts a group normalization layer

Adds a group normalization layer separating the input into `<groups>` groups,
normalizing each separately.
