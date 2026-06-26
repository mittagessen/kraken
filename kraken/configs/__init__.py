from .ro import *  # NOQA
from .base import *  # NOQA
from .vgsl import *  # NOQA
from .pretrain import *  # NOQA

import torch.serialization
from collections import defaultdict

from .base import __all__ as _base_all
from .vgsl import __all__ as _vgsl_all
from .ro import __all__ as _ro_all
from .pretrain import __all__ as _pretrain_all

from .base import _Counter

torch.serialization.add_safe_globals([globals()[name] for name in _base_all + _vgsl_all + _ro_all + _pretrain_all] +
                                     [defaultdict, _Counter])
