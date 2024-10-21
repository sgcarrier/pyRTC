
from pyRTC.SlopesProcess import *
from pyRTC.Pipeline import *
from pyRTC.utils import *
import argparse
import sys
import os 


class NoisySlopesProcess(SlopesProcess):

    def __init__(self, conf):
        super().__init__(conf)

        self.total_photon_flux = 100000



        