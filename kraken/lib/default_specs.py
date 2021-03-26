# -*- coding: utf-8 -*-
#
# Copyright 2020 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
Default VGSL specs and hyperparameters
"""

SEGMENTATION_SPEC = '[1,1200,0,3 Cr7,7,64,2,2 Gn32 Cr3,3,128,2,2 Gn32 Cr3,3,128 Gn32 Cr3,3,256 Gn32 Cr3,3,256 Gn32 Lbx32 Lby32 Cr1,1,32 Gn32 Lby32 Lbx32]'
RECOGNITION_SPEC = '[1,48,0,1 Cr4,2,32,4,2 Gn32 Cr4,2,64,1,1 Gn32 Mp4,2,4,2 Cr3,3,128,1,1 Gn32 Mp1,2,1,2 S1(1x0)1,3 Lbx256 Do0.5 Lbx256 Do0.5 Lbx256 Do0.5]'

RECOGNITION_HYPER_PARAMS = {'pad': 16,
                            'freq': 1.0,
                            'batch_size': 1,
                            'quit': 'early',
                            'epochs': -1,
                            'lag': 5,
                            'min_delta': None,
                            'optimizer': 'Adam',
                            'lrate': 1e-3,
                            'momentum': 0.9,
                            'weight_decay': 0,
                            'schedule': 'constant',
                            'normalization': None,
                            'normalize_whitespace': True,
                            'completed_epochs': 0,
                            'augment': False,
                            # lr scheduler params
                            # step/exp decay
                            'step_size': 10,
                            'gamma': 0.1,
                            # reduce on plateau
                            'rop_patience': 5,
                            # cosine
                            'cos_t_max': 50,
                           }

SEGMENTATION_HYPER_PARAMS = {'line_width': 8,
                             'freq': 1.0,
                             'quit': 'dumb',
                             'epochs': 50,
                             'lag': 10,
                             'min_delta': None,
                             'optimizer': 'Adam',
                             'lrate': 2e-4,
                             'momentum': 0.9,
                             'weight_decay': 1e-5,
                             'schedule': 'constant',
                             'completed_epochs': 0,
                             'augment': False,
                            # lr scheduler params
                            # step/exp decay
                            'step_size': 10,
                            'gamma': 0.1,
                            # reduce on plateau
                            'rop_patience': 5,
                            # cosine
                            'cos_t_max': 50,

                           }
