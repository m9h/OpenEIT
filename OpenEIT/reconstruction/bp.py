"""

# Copyright (c) Mindseye Biomedical LLC. All rights reserved.
# Distributed under the (new) CC BY-NC-SA 4.0 License. See LICENSE.txt for more info.


This module contains an image reconstruction implementation based
on interpolation and inverse radon transformation.

"""

import logging
import itertools
import time
import threading

import numpy as np
from pyeit import mesh
from pyeit.eit import protocol
from pyeit.eit.bp import BP as bp

logger = logging.getLogger(__name__)


class BpReconstruction:

    """
    Reconstruction of image data from an EIT measurement.
    Using old fashioned Back Projection. 

    """
    def __init__(self,n_el):
        # setup EIT scan conditions
        self.img = []
        self.baseline_flag = 0
        self.n_el = n_el # number of electrodes. 
        self.step = 1 
        self.el_dist = int(self.n_el/2) # random initialize number 

        # we create this according to an opposition protocol to maximize contrast. 
        # opposition drive, adjacent measurement: the pattern the firmware scans
        self.protocol = protocol.create(self.n_el, dist_exc=self.el_dist, step_meas=self.step, parser_meas='std')
        self.ex_mat = self.protocol.ex_mat
        """ 0. construct mesh """
        # h0 is initial mesh size. , h0=0.1
        self.mesh_obj = mesh.create(self.n_el)
        self.el_pos = self.mesh_obj.el_pos
        """ 3. Set Up BP """
        self.eit = bp(self.mesh_obj, self.protocol)

        self.eit.setup(weight='none')

    def update_reference(self,data):
        self.baseline_flag = 1

    def eit_reconstruction(self, data):
        """
        Reconstruct an image from the measurements given by `data`.

        """
        try: 
            if self.baseline_flag == 1:
                self.f0 = np.array(data)
                self.baseline_flag = 0 
            # data contains fl.v and f0.v 
            f1 = np.array(data)
            # if the jacobian is not normalized, data may not to be normalized too.
            ds_bp = self.eit.solve(f1, self.f0, normalize=True)
            self.img = np.real(ds_bp)

        except RuntimeError as err:
            logger.error('reconstruction problem: %s', err)

        return self.img
